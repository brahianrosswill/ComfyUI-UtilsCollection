import re
import torch
import math
import os

from comfy_api.latest import ComfyExtension, io
from comfy.utils import common_upscale
import node_helpers
from .helper_functions import get_token_count, get_token_count_scaled

class UC_AttentionBiasTextEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_AttentionBiasTextEncode",
            category="advanced/conditioning",
            display_name="CLIP Text Encode with Attention Bias (Experimental)",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("text", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
            ]
        )

    @classmethod
    def execute(cls, clip, text) -> io.NodeOutput:
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        if '<' not in text and '>' not in text and '=' not in text:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            return ([[cond, {"pooled_output": pooled}]], )

        bias_pattern = re.compile(r"<([^>]+)=([0-9.-]+)>")
        split_pattern = re.compile(r"(<[^>]+=[0-9.-]+>)")
        segments = split_pattern.split(text)

        clean_text = ""
        biases_to_apply = []

        current_token_index = 1

        for segment in segments:
            if not segment:
                continue

            match = bias_pattern.fullmatch(segment)
            if match:
                bias_text, strength_str = match.groups()
                strength = float(strength_str)
                clean_text += bias_text
                num_tokens = get_token_count(clip, bias_text)

                if num_tokens > 0:
                    start_index = current_token_index
                    end_index = current_token_index + num_tokens
                    biases_to_apply.append({"start": start_index, "end": end_index, "strength": strength})

                current_token_index += num_tokens
            else:
                clean_text += segment
                num_tokens = get_token_count(clip, segment)
                current_token_index += num_tokens

        tokens = clip.tokenize(clean_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        if not biases_to_apply:
            return ([[cond, {"pooled_output": pooled}]], )

        cond_dict = {"pooled_output": pooled}
        n_text_tokens = cond.shape[1]
        device = cond.device
        dtype = torch.float16


        final_seq_len = n_text_tokens + 1
        attn_mask = torch.zeros((1, final_seq_len, final_seq_len), dtype=dtype, device=device)

        pooled_offset = 1

        for bias in biases_to_apply:
            strength = bias["strength"]
            attn_bias_value = torch.log(torch.tensor(strength, dtype=dtype, device=device))

            start = min(bias["start"] + pooled_offset, final_seq_len)
            end = min(bias["end"] + pooled_offset, final_seq_len)

            if start >= end:
                continue

            attn_mask[:, :, start:end] += attn_bias_value
            attn_mask[:, start:end, :] += attn_bias_value

        cond_dict["attention_mask"] = attn_mask
        cond_dict["attention_mask_img_shape"] = (1, 1)


        new_conditioning = ([[cond, cond_dict]])

        return io.NodeOutput(new_conditioning)

def encode_embedding_scaled_bias(clip, text, llama_template=None, **kwargs):
    if clip is None:
        raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

    if "<" not in text and ">" not in text and "=" not in text:
        tokens = clip.tokenize(text, llama_template=llama_template, **kwargs)
        return clip.encode_from_tokens_scheduled(tokens)

    # Permissive regex for whitespace inside tags
    bias_pattern = re.compile(r"<\s*([^>=]+?)\s*=\s*([0-9.-]+)\s*>")
    split_pattern = re.compile(r"(<\s*[^>=]+?\s*=\s*[0-9.-]+\s*>)")
    segments = split_pattern.split(text)

    clean_text = ""
    biases_to_apply = []

    # Use prefix-only template for measurements to avoid suffix-induced shifts
    prefix_template = "{}"
    if llama_template:
        prefix_template = llama_template.split("{}")[0] + "{}"

    for segment in segments:
        if not segment:
            continue

        match = bias_pattern.fullmatch(segment)
        if match:
            # Count before adding biased segment
            start_count = get_token_count_scaled(clip, clean_text, llama_template=prefix_template)

            bias_text, strength_str = match.groups()
            clean_text += bias_text

            # Count after adding biased segment
            end_count = get_token_count_scaled(clip, clean_text, llama_template=prefix_template)

            if end_count > start_count:
                # BOS is at index 0, so tokens are at indices 1 to count
                start_index = 1 + start_count
                end_index = 1 + end_count
                biases_to_apply.append({"start": start_index, "end": end_index, "strength": float(strength_str)})
        else:
            clean_text += segment

    tokens = clip.tokenize(clean_text, llama_template=llama_template, **kwargs)
    conditioning = clip.encode_from_tokens_scheduled(tokens)

    if not biases_to_apply:
        return conditioning

    # Apply bias scaling directly to each schedule in conditioning
    new_conditioning = []
    max_strength = 1.0
    for bias in biases_to_apply:
        max_strength = max(max_strength, bias["strength"])

    for i in range(len(conditioning)):
        cond, cond_dict = conditioning[i]

        # Directly scale the embeddings for the biased tokens
        new_cond = cond.clone()

        for bias in biases_to_apply:
            strength = bias["strength"]
            start = min(bias["start"], new_cond.shape[1])
            end = min(bias["end"], new_cond.shape[1])

            if start >= end:
                continue

            new_cond[:, start:end, :] *= strength

        new_cond_dict = cond_dict.copy()
        if "pooled_output" in new_cond_dict and new_cond_dict["pooled_output"] is not None:
            new_cond_dict["pooled_output"] = new_cond_dict["pooled_output"].clone() * max_strength

        new_conditioning.append([new_cond, new_cond_dict])

    return new_conditioning

from enum import Enum

class ImageInputMapping(Enum):
    ZERO_INDEXED_OFFSET = 1
    ONE_INDEXED_OFFSET = 0

    @classmethod
    def get_display_num(cls, num, is_zero_indexed):
        offset = cls.ZERO_INDEXED_OFFSET.value if is_zero_indexed else cls.ONE_INDEXED_OFFSET.value
        return num + offset

    @classmethod
    def get_display_name(cls, num, is_zero_indexed):
        return f"image_input_{cls.get_display_num(num, is_zero_indexed)}"

    @classmethod
    def get_dict_key(cls, num, is_zero_indexed):
        offset = cls.ZERO_INDEXED_OFFSET.value if is_zero_indexed else cls.ONE_INDEXED_OFFSET.value
        return num - offset

def is_image_token(t):
    if isinstance(t, tuple) and len(t) > 0:
        val = t[0]
    else:
        val = t

    if isinstance(val, dict) and val.get("type") == "image":
        return True

    if val in (151655, 262144): # Qwen & Gemma image pad IDs
        return True

    return False

def evaluate_formula(expression: str, processed_images: dict) -> torch.Tensor:
    # Sandboxed evaluation dictionary
    safe_dict = {
        "__builtins__": {},
        "clamp": torch.clamp,
        "min": torch.minimum,
        "max": torch.maximum,
        "abs": torch.abs,
    }

    # Inject variables
    for name, tensor in processed_images.items():
        safe_dict[name] = tensor

    try:
        # PyTorch overloads +, -, *, /, ** on tensors automatically!
        result = eval(expression, safe_dict, {})  # noqa: S307

        # Ensure result stays bounded between 0.0 and 1.0 (clamping standard image pixel range)
        return torch.clamp(result, 0.0, 1.0)
    except Exception as e:
        raise RuntimeError(f"Error evaluating visual math expression '|{expression}|': {e}")

def evaluate_conditioning_formula(expression: str, sequence_tensors: dict, pooled_tensors: dict, padding_method: str = "zero-pad") -> tuple:
    # Preprocess classic weighting syntax inside math expression to math scaling
    # e.g., (image_input_1:10) -> (image_input_1 * 10)
    expression = re.sub(
        r"\(\s*([a-zA-Z0-9_]+)\s*:\s*([0-9.-]+)\s*\)",
        r"(\1 * \2)",
        expression
    )

    # Determine max sequence length across all tensors
    max_len = max(tensor.shape[1] for tensor in sequence_tensors.values())

    # Pad/interpolate all tensors to match max length exactly
    aligned_sequence_tensors = {}
    for name, tensor in sequence_tensors.items():
        if tensor.shape[1] < max_len:
            if padding_method == "interpolate":
                import torch.nn.functional as F
                tensor_perm = tensor.permute(0, 2, 1)
                tensor_interp = F.interpolate(tensor_perm, size=max_len, mode='linear', align_corners=False)
                tensor = tensor_interp.permute(0, 2, 1)
            else: # zero-pad
                pad_size = max_len - tensor.shape[1]
                padding = torch.zeros((tensor.shape[0], pad_size, tensor.shape[2]), device=tensor.device, dtype=tensor.dtype)
                tensor = torch.cat([tensor, padding], dim=1)
        aligned_sequence_tensors[name] = tensor

    safe_dict_cond = {
        "__builtins__": {},
        "clamp": torch.clamp,
        "min": torch.minimum,
        "max": torch.maximum,
        "abs": torch.abs,
    }
    for name, tensor in aligned_sequence_tensors.items():
        safe_dict_cond[name] = tensor

    safe_dict_pooled = {
        "__builtins__": {},
        "clamp": torch.clamp,
        "min": torch.minimum,
        "max": torch.maximum,
        "abs": torch.abs,
    }
    for name, tensor in pooled_tensors.items():
        if tensor is not None:
            safe_dict_pooled[name] = tensor

    try:
        C_blended = eval(expression, safe_dict_cond, {})  # noqa: S307
        P_blended = None
        if any(v is not None for v in pooled_tensors.values()):
            P_blended = eval(expression, safe_dict_pooled, {})  # noqa: S307
        return C_blended, P_blended
    except Exception as e:
        raise RuntimeError(f"Error evaluating conditioning math expression '{expression}': {e}")

def find_visual_token_range(tokens, cond_tensor) -> tuple:
    # Build dynamic mapping from tokens to embeddings
    key_name = next(iter(tokens.keys()))
    token_list = tokens[key_name][0]

    text_token_count = 0
    image_token_count = 0
    for t in token_list:
        if is_image_token(t):
            image_token_count += 1
        else:
            text_token_count += 1

    if image_token_count == 0:
        return 0, 0

    V = (cond_tensor.shape[1] - text_token_count) // image_token_count

    mapping = []
    current_idx = 0
    for t in token_list:
        is_img = is_image_token(t)
        size = V if is_img else 1
        start = current_idx
        end = current_idx + size
        mapping.append((start, end))
        current_idx = end

    # Find the first visual token index range
    for i, t in enumerate(token_list):
        if is_image_token(t):
            return mapping[i][0], mapping[i][1]

    return 0, 0

def encode_embedding_classical_scaled_bias(clip, text, llama_template=None, **kwargs):
    if clip is None:
        raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

    if "(" not in text or ":" not in text or ")" not in text:
        tokens = clip.tokenize(text, llama_template=llama_template, **kwargs)
        return clip.encode_from_tokens_scheduled(tokens)

    # Regex for classical weighting syntax, e.g., (blue sky:1.2) or (sunset:0.8)
    bias_pattern = re.compile(r"\(\s*([^:)]+?)\s*:\s*([0-9.-]+)\s*\)")
    split_pattern = re.compile(r"(\(\s*[^:)]+?\s*:\s*[0-9.-]+\s*\))")
    segments = split_pattern.split(text)

    clean_text = ""
    biases_to_apply = []

    for segment in segments:
        if not segment:
            continue

        match = bias_pattern.fullmatch(segment)
        if match:
            # Measure token length of clean text before appending bias text
            start_tokens = clip.tokenize(clean_text, llama_template=llama_template, **kwargs)
            key_name = next(iter(start_tokens.keys()))
            start_count = len(start_tokens[key_name][0])

            bias_text, strength_str = match.groups()
            clean_text += bias_text

            # Measure token length of clean text after appending bias text
            end_tokens = clip.tokenize(clean_text, llama_template=llama_template, **kwargs)
            end_count = len(end_tokens[key_name][0])

            if end_count > start_count:
                biases_to_apply.append({"start": start_count, "end": end_count, "strength": float(strength_str)})
        else:
            clean_text += segment

    tokens = clip.tokenize(clean_text, llama_template=llama_template, **kwargs)
    conditioning = clip.encode_from_tokens_scheduled(tokens)

    if not biases_to_apply:
        return conditioning

    # Apply bias scaling directly to each schedule in conditioning
    new_conditioning = []
    max_strength = 1.0
    for bias in biases_to_apply:
        max_strength = max(max_strength, bias["strength"])

    for i in range(len(conditioning)):
        cond, cond_dict = conditioning[i]
        new_cond = cond.clone()

        # Build dynamic mapping from tokens to embeddings
        key_name = next(iter(tokens.keys()))
        token_list = tokens[key_name][0]

        text_token_count = 0
        image_token_count = 0
        for t in token_list:
            if is_image_token(t):
                image_token_count += 1
            else:
                text_token_count += 1

        # Calculate expansion factor V
        if image_token_count > 0:
            V = (new_cond.shape[1] - text_token_count) // image_token_count
        else:
            V = 1

        # Build start/end embedding index mapping for each token
        mapping = []
        current_idx = 0
        for t in token_list:
            is_img = is_image_token(t)
            size = V if is_img else 1
            start = current_idx
            end = current_idx + size
            mapping.append((start, end))
            current_idx = end

        # Scale embeddings using mapped ranges
        for bias in biases_to_apply:
            strength = bias["strength"]

            # Map token indices to embedding indices
            t_start = bias["start"]
            t_end = bias["end"]

            if t_start >= len(mapping):
                continue

            start = mapping[t_start][0]
            end = mapping[min(t_end - 1, len(mapping) - 1)][1]

            if start >= end:
                continue

            new_cond[:, start:end, :] *= strength

        new_cond_dict = cond_dict.copy()
        if "pooled_output" in new_cond_dict and new_cond_dict["pooled_output"] is not None:
            new_cond_dict["pooled_output"] = new_cond_dict["pooled_output"].clone() * max_strength

        new_conditioning.append([new_cond, new_cond_dict])

    return new_conditioning

class UC_ScaledBiasTextEncodeFlux2SystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ScaledBiasTextEncodeFlux2SystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Flux2 dev System Prompt (Scaled Bias)",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt=None) -> io.NodeOutput:
        if len(system_prompt) > 0:
            template_prefix = r"[SYSTEM_PROMPT]"
            template_suffix = r"[/SYSTEM_PROMPT][INST]{}[/INST]"
            llama_template = f"{template_prefix}{system_prompt}{template_suffix}"
            conditioning = encode_embedding_scaled_bias(clip, prompt, llama_template=llama_template)
        else:
            conditioning = encode_embedding_scaled_bias(clip, prompt)

        return io.NodeOutput(conditioning)


class UC_ScaledBiasTextEncodeKleinSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ScaledBiasTextEncodeKleinSystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Flux2 Klein System Prompt (Scaled Bias)",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.String.Input(
                    "thinking_content",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip="Custom thinking content to inject. Leave empty for default.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt="", thinking_content="") -> io.NodeOutput:
        # Build template with string concat (ComfyUI pattern)
        if len(system_prompt) > 0:
            llama_template = (
                "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                "<|im_start|>user\n{}<|im_end|>\n" +
                "<|im_start|>assistant\n<think>\n" + thinking_content + "\n</think>\n\n"
            )
        else:
            llama_template = (
                "<|im_start|>user\n{}<|im_end|>\n" +
                "<|im_start|>assistant\n<think>\n" + thinking_content + "\n</think>\n\n"
            )

        conditioning = encode_embedding_scaled_bias(clip, prompt, llama_template=llama_template)
        return io.NodeOutput(conditioning)


class UC_ScaledBiasTextEncodeLtxv2SystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ScaledBiasTextEncodeLtxv2SystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with LTXV 2 System Prompt (Scaled Bias)",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Image.Input("image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt="", image=None) -> io.NodeOutput:
        # Build template with string concat (ComfyUI pattern)
        if image is not None:
            llama_template = (
                "<start_of_turn>system\n" + system_prompt + "<end_of_turn>\n" +
                "<start_of_turn>user\n\n<image_soft_token>{}<end_of_turn>\n\n<start_of_turn>model\n"
            )
        elif len(system_prompt) > 0:
            llama_template = (
                "<start_of_turn>system\n" + system_prompt + "<end_of_turn>\n" +
                "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
            )
        else:
            llama_template = (
                "<start_of_turn>system\nYou are a helpful assistant.<end_of_turn>\n<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
            )

        conditioning = encode_embedding_scaled_bias(clip, prompt, llama_template=llama_template, image=image)
        return io.NodeOutput(conditioning)


class UC_ScaledBiasTextEncodeZITSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ScaledBiasTextEncodeZITSystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Z-Image System Prompt (Scaled Bias)",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt=None) -> io.NodeOutput:
        if len(system_prompt) > 0:
            template_prefix = "<|im_start|>system\n"
            template_suffix = (
                "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            )
            llama_template = f"{template_prefix}{system_prompt}{template_suffix}"
            conditioning = encode_embedding_scaled_bias(clip, prompt, llama_template=llama_template)
        else:
            conditioning = encode_embedding_scaled_bias(clip, prompt)

        return io.NodeOutput(conditioning)


class UC_ScaledBiasTextEncodeZImageThinkPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ScaledBiasTextEncodeZImageThinkPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Z-Image Thinking Prompt (Scaled Bias)",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("thinking", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, thinking=None) -> io.NodeOutput:
        if len(thinking) > 0:
            template_prefix = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            template_suffix = "\n</think>\n\n"
            llama_template = f"{template_prefix}{thinking}{template_suffix}"
            conditioning = encode_embedding_scaled_bias(clip, prompt, llama_template=llama_template)
        else:
            conditioning = encode_embedding_scaled_bias(clip, prompt)

        return io.NodeOutput(conditioning)


class UC_ScaledBiasTextEncodeSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ScaledBiasTextEncodeSystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode System Prompt (Scaled Bias)",
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input(
                    "model_type",
                    options=["flux2dev", "klein", "z-image"],
                    default="flux2dev",
                    tooltip="Select the model type to use the correct template format.",
                ),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.String.Input(
                    "thinking_content",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip="(Klein only) Custom thinking content to inject. Leave empty for default.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, model_type, prompt, system_prompt="", thinking_content="") -> io.NodeOutput:
        if model_type == "klein" and len(thinking_content) > 0:
            # Klein with custom thinking content
            if len(system_prompt) > 0:
                llama_template = (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n" +
                    f"<|im_start|>user\n{{}}<|im_end|>\n" +
                    f"<|im_start|>assistant\n<think>\n{thinking_content}\n</think>\n\n"
                )
            else:
                llama_template = (
                    "<|im_start|>user\n{}<|im_end|>\n" +
                    f"<|im_start|>assistant\n<think>\n{thinking_content}\n</think>\n\n"
                )
        elif len(system_prompt) > 0:
            template = SYSTEM_PROMPT_TEMPLATES.get(model_type, SYSTEM_PROMPT_TEMPLATES["flux2dev"])
            llama_template = f"{template['prefix']}{system_prompt}{template['suffix']}"
        else:
            llama_template = None

        conditioning = encode_embedding_scaled_bias(clip, prompt, llama_template=llama_template)
        return io.NodeOutput(conditioning)


class UC_TextEncodeFlux2SystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeFlux2SystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Flux2 dev System Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt=None) -> io.NodeOutput:
        if len(system_prompt) > 0:
            template_prefix = r"[SYSTEM_PROMPT]"
            template_suffix = r"[/SYSTEM_PROMPT][INST]{}[/INST]"
            llama_template = f"{template_prefix}{system_prompt}{template_suffix}"
            tokens = clip.tokenize(prompt, llama_template=llama_template)
        else:
            tokens = clip.tokenize(prompt)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class UC_TextEncodeKleinSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeKleinSystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Flux2 Klein System Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.String.Input(
                    "thinking_content",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip="Custom thinking content to inject. Leave empty for default.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt="", thinking_content="") -> io.NodeOutput:
        # Build template with string concat (ComfyUI pattern)
        if len(system_prompt) > 0:
            llama_template = (
                "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                "<|im_start|>user\n{}<|im_end|>\n" +
                "<|im_start|>assistant\n<think>\n" + thinking_content + "\n</think>\n\n"
            )
        else:
            llama_template = (
                "<|im_start|>user\n{}<|im_end|>\n" +
                "<|im_start|>assistant\n<think>\n" + thinking_content + "\n</think>\n\n"
            )

        tokens = clip.tokenize(prompt, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class UC_TextEncodeKrea2SystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeKrea2SystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Krea2 System Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt="") -> io.NodeOutput:
        # Build template with string concat (ComfyUI pattern)
        if len(system_prompt) > 0:
            llama_template = (
                "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                "<|im_start|>user\n{}<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )
        else:
            llama_template = (
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
                "<|im_start|>user\n{}<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )

        tokens = clip.tokenize(prompt, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class TextEncodeSystemEditPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeSystemEditPlus",
            display_name="TextEncodeSystemEditPlus",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path). 'Fast' = 384x384, 'Balanced' = 512x512, 'Detailed' = 768x768, 'Large' = 1024x1024, 'X-Large' = 1280x1280, 'Original' uses native resolution.",
                ),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path). 'Fast' = 1024x1024, 'Balanced' = 1280x1280, 'Detailed' = 1536x1536, 'Original' uses native resolution.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt, vlm_resolution, vae_resolution, vae=None, image1=None, image2=None, image3=None) -> io.NodeOutput:
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        image_prompt = ""

        VLM_RESOLUTIONS = {
            "Fast (384)": 384,
            "Balanced (512)": 512,
            "Detailed (768)": 768,
            "Large (1024)": 1024,
            "X-Large (1280)": 1280,
            "XX-Large (1536)": 1536
        }

        VAE_RESOLUTIONS = {
            "Ultra (512)": 512,
            "Turbo (768)": 768,
            "Fast (1024)": 1024,
            "Balanced (1280)": 1280,
            "Detailed (1536)": 1536
        }

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)

                # 1. Semantic Path Scaling (VLM)
                if vlm_resolution == "Original":
                    images_vl.append(image)
                else:
                    vlm_size = VLM_RESOLUTIONS[vlm_resolution]
                    total_vlm = vlm_size * vlm_size
                    scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                    width_vlm = round(samples.shape[3] * scale_by_vlm)
                    height_vlm = round(samples.shape[2] * scale_by_vlm)

                    s_vlm = common_upscale(samples, width_vlm, height_vlm, "bicubic", "disabled")
                    images_vl.append(s_vlm.movedim(1, -1))

                # 2. Structural Path Scaling (VAE)
                if vae is not None:
                    if vae_resolution == "Original":
                        width_vae = round(samples.shape[3] / 8.0) * 8
                        height_vae = round(samples.shape[2] / 8.0) * 8
                        s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                        ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))
                    else:
                        vae_size = VAE_RESOLUTIONS[vae_resolution]
                        total_vae = vae_size * vae_size
                        scale_by_vae = math.sqrt(total_vae / (samples.shape[3] * samples.shape[2]))
                        width_vae = round(samples.shape[3] * scale_by_vae / 8.0) * 8
                        height_vae = round(samples.shape[2] * scale_by_vae / 8.0) * 8

                        s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                        ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        # Construct the complete template string via safe concatenation to prevent formatting errors
        if len(system_prompt) > 0:
            full_prompt = (
                "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                "<|im_start|>user\n" + image_prompt + prompt + "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )
        else:
            full_prompt = (
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
                "<|im_start|>user\n" + image_prompt + prompt + "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )

        # Pass skip_template=True so the tokenizer doesn't try to wrap or append extra blocks
        tokens = clip.tokenize(full_prompt, images=images_vl, skip_template=True)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        return io.NodeOutput(conditioning)


class TextEncodeSystemEditPlusAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="TextEncodeSystemEditPlusAdvanced",
            display_name="TextEncodeSystemEditPlusAdvanced",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path). 'Fast' = 384x384, 'Balanced' = 512x512, 'Detailed' = 768x768, 'Large' = 1024x1024, 'X-Large' = 1280x1280, 'XX-Large' = 1536x1536, 'Original' uses native resolution.",
                ),
                io.Autogrow.Input("image_inputs", template=autogrow_template),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt, vlm_resolution, image_inputs: io.Autogrow.Type) -> io.NodeOutput:
        # Collect and parse all autogrow keys
        raw_images = {}
        if image_inputs is not None:
            # image_inputs can be a dict, let's handle cases where it might be empty
            for k, v in image_inputs.items():
                if v is not None:
                    # Extract numeric suffix (e.g. "image1" -> 1)
                    digits = re.findall(r'\d+', k)
                    if digits:
                        idx = int(digits[0])
                    else:
                        idx = 1
                    raw_images[idx] = v

        # Determine indexing: 0-indexed or 1-indexed.
        is_zero_indexed = 0 in raw_images

        # Check if the prompt has any image_input_ keyword matches (case-insensitive)
        pattern = re.compile(r'image_input_(\d+)', re.IGNORECASE)

        images_vl = []

        def process_vlm_image(image, res):
            if image is None:
                return None
            VLM_RESOLUTIONS = {
                "Fast (384)": 384,
                "Balanced (512)": 512,
                "Detailed (768)": 768,
                "Large (1024)": 1024,
                "X-Large (1280)": 1280,
                "XX-Large (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            if res == "Original":
                return image
            else:
                vlm_size = VLM_RESOLUTIONS[res]
                total_vlm = vlm_size * vlm_size
                scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                width_vlm = round(samples.shape[3] * scale_by_vlm)
                height_vlm = round(samples.shape[2] * scale_by_vlm)

                s_vlm = common_upscale(samples, width_vlm, height_vlm, "bicubic", "disabled")
                return s_vlm.movedim(1, -1)

        # Create dict of preprocessed VLM images for math evaluation
        processed_images = {}
        for num, img in raw_images.items():
            name = ImageInputMapping.get_display_name(num, is_zero_indexed)
            processed_images[name] = process_vlm_image(img, vlm_resolution)

        # Parse and replace any math formulas enclosed in pipes |formula|
        math_pattern = re.compile(r"\|([^|]+)\|")

        def replace_formula(match):
            expression = match.group(1).strip()
            result_tensor = evaluate_formula(expression, processed_images)
            images_vl.append(result_tensor)
            return "<|vision_start|><|image_pad|><|vision_end|>"

        modified_prompt = math_pattern.sub(replace_formula, prompt)

        # Re-check for keywords in the modified prompt
        has_keywords = bool(pattern.search(modified_prompt)) or len(images_vl) > 0

        if has_keywords:
            # Replace keywords dynamically and build images_vl in order of appearance
            def replace_keyword(match):
                num = int(match.group(1))
                dict_key = ImageInputMapping.get_dict_key(num, is_zero_indexed)
                if dict_key in raw_images:
                    img = raw_images[dict_key]
                    processed_img = processed_images.get(ImageInputMapping.get_display_name(dict_key, is_zero_indexed), process_vlm_image(img, vlm_resolution))
                    images_vl.append(processed_img)
                    return "<|vision_start|><|image_pad|><|vision_end|>"
                return ""

            modified_prompt = pattern.sub(replace_keyword, modified_prompt)
        else:
            # Fallback: prepend all connected images in numerical order of their slots
            image_prompt = ""
            for num in sorted(raw_images.keys()):
                name = ImageInputMapping.get_display_name(num, is_zero_indexed)
                processed_img = processed_images[name]
                images_vl.append(processed_img)
                image_prompt += f"<|vision_start|><|image_pad|><|vision_end|>"

            modified_prompt = image_prompt + modified_prompt

        # Construct the complete template string via safe concatenation
        if len(system_prompt) > 0:
            full_prompt = (
                "<|im_start|>user\n" + "<|im_end|>\n" +
                "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                "<|im_start|>user\n" + modified_prompt + "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )
        else:
            full_prompt = (
                "<|im_start|>user\n" + "<|im_end|>\n" +
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
                "<|im_start|>user\n" + modified_prompt + "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )

        # Pass skip_template=True so the tokenizer doesn't try to wrap or append extra blocks
        tokens = clip.tokenize(full_prompt, images=images_vl, skip_template=True)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class TextEncodeKrea2SystemEditPlusAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="TextEncodeKrea2SystemEditPlusAdvanced",
            display_name="TextEncodeKrea2SystemEditPlusAdvanced",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip=(
                        "Main text prompt. Supports visual math blending: |formula| to blend image inputs at pixel-tensor level before encoding. "
                        "Example: |((image_input_1 * 1.075) + (image_input_2 * 1.025)) / 1.5| to blend styles/concepts. "
                        "Supported math operations: +, -, *, /, clamp, min, max, abs, on variables image_input_1 to image_input_16."
                    ),
                ),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path). 'Fast' = 384x384, 'Balanced' = 512x512, 'Detailed' = 768x768, 'Large' = 1024x1024, 'X-Large' = 1280x1280, 'XX-Large' = 1536x1536, 'Original' uses native resolution.",
                ),
                io.Autogrow.Input("image_inputs", template=autogrow_template),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt, vlm_resolution, image_inputs: io.Autogrow.Type) -> io.NodeOutput:
        # Collect and parse all autogrow keys
        raw_images = {}
        if image_inputs is not None:
            # image_inputs can be a dict, let's handle cases where it might be empty
            for k, v in image_inputs.items():
                if v is not None:
                    # Extract numeric suffix (e.g. "image1" -> 1)
                    digits = re.findall(r'\d+', k)
                    if digits:
                        idx = int(digits[0])
                    else:
                        idx = 1
                    raw_images[idx] = v

        # Determine indexing: 0-indexed or 1-indexed.
        is_zero_indexed = 0 in raw_images

        # Check if the prompt has any image_input_ keyword matches (case-insensitive)
        pattern = re.compile(r'image_input_(\d+)', re.IGNORECASE)

        images_vl = []

        def process_vlm_image(image, res):
            if image is None:
                return None
            VLM_RESOLUTIONS = {
                "Fast (384)": 384,
                "Balanced (512)": 512,
                "Detailed (768)": 768,
                "Large (1024)": 1024,
                "X-Large (1280)": 1280,
                "XX-Large (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            if res == "Original":
                return image
            else:
                vlm_size = VLM_RESOLUTIONS[res]
                total_vlm = vlm_size * vlm_size
                scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                width_vlm = round(samples.shape[3] * scale_by_vlm)
                height_vlm = round(samples.shape[2] * scale_by_vlm)

                s_vlm = common_upscale(samples, width_vlm, height_vlm, "bicubic", "disabled")
                return s_vlm.movedim(1, -1)

        # Create dict of preprocessed VLM images for math evaluation
        processed_images = {}
        for num, img in raw_images.items():
            name = ImageInputMapping.get_display_name(num, is_zero_indexed)
            processed_images[name] = process_vlm_image(img, vlm_resolution)

        # Parse and replace any math formulas enclosed in pipes |formula|
        math_pattern = re.compile(r"\|([^|]+)\|")

        def replace_formula(match):
            expression = match.group(1).strip()
            result_tensor = evaluate_formula(expression, processed_images)
            images_vl.append(result_tensor)
            return "<|vision_start|><|image_pad|><|vision_end|>"

        modified_prompt = math_pattern.sub(replace_formula, prompt)

        # Re-check for keywords in the modified prompt
        has_keywords = bool(pattern.search(modified_prompt)) or len(images_vl) > 0

        if has_keywords:
            # Replace keywords dynamically and build images_vl in order of appearance
            def replace_keyword(match):
                num = int(match.group(1))
                dict_key = ImageInputMapping.get_dict_key(num, is_zero_indexed)
                if dict_key in raw_images:
                    img = raw_images[dict_key]
                    processed_img = processed_images.get(ImageInputMapping.get_display_name(dict_key, is_zero_indexed), process_vlm_image(img, vlm_resolution))
                    images_vl.append(processed_img)
                    return "<|vision_start|><|image_pad|><|vision_end|>"
                return ""

            modified_prompt = pattern.sub(replace_keyword, modified_prompt)
        else:
            # Fallback: prepend all connected images in numerical order of their slots
            image_prompt = ""
            for num in sorted(raw_images.keys()):
                name = ImageInputMapping.get_display_name(num, is_zero_indexed)
                processed_img = processed_images[name]
                images_vl.append(processed_img)
                image_prompt += f"<|vision_start|><|image_pad|><|vision_end|>"

            modified_prompt = image_prompt + modified_prompt

        # Construct the complete template string via safe concatenation
        if len(system_prompt) > 0:
            full_prompt = (
                "<|im_start|>user\n" + "<|im_end|>\n" +
                "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                "<|im_start|>user\n" + modified_prompt + "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )
        else:
            full_prompt = (
                "<|im_start|>user\n" + "<|im_end|>\n" +
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
                "<|im_start|>user\n" + modified_prompt + "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )

        # Pass skip_template=True so the tokenizer doesn't try to wrap or append extra blocks
        tokens = clip.tokenize(full_prompt, images=images_vl, skip_template=True)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class TextEncodeEditPlusAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="TextEncodeEditPlusAdvanced",
            display_name="TextEncodeEditPlusAdvanced",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip=(
                        "Main text prompt. Supports visual math blending: |formula| to blend image inputs at pixel-tensor level before encoding. "
                        "Example: |((image_input_1 * 1.075) + (image_input_2 * 1.025)) / 1.5| to blend styles/concepts. "
                        "Supported math operations: +, -, *, /, clamp, min, max, abs, on variables image_input_1 to image_input_16."
                    ),
                ),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path). 'Fast' = 384x384, 'Balanced' = 512x512, 'Detailed' = 768x768, 'Large' = 1024x1024, 'X-Large' = 1280x1280, 'XX-Large' = 1536x1536, 'Original' uses native resolution.",
                ),
                io.Autogrow.Input("image_inputs", template=autogrow_template),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vlm_resolution, image_inputs: io.Autogrow.Type) -> io.NodeOutput:
        # Collect and parse all autogrow keys
        raw_images = {}
        if image_inputs is not None:
            # image_inputs can be a dict, let's handle cases where it might be empty
            for k, v in image_inputs.items():
                if v is not None:
                    # Extract numeric suffix (e.g. "image1" -> 1)
                    digits = re.findall(r'\d+', k)
                    if digits:
                        idx = int(digits[0])
                    else:
                        idx = 1
                    raw_images[idx] = v

        # Determine indexing: 0-indexed or 1-indexed.
        is_zero_indexed = 0 in raw_images

        # Check if the prompt has any image_input_ keyword matches (case-insensitive)
        pattern = re.compile(r'image_input_(\d+)', re.IGNORECASE)

        images_vl = []

        def process_vlm_image(image, res):
            if image is None:
                return None
            VLM_RESOLUTIONS = {
                "Fast (384)": 384,
                "Balanced (512)": 512,
                "Detailed (768)": 768,
                "Large (1024)": 1024,
                "X-Large (1280)": 1280,
                "XX-Large (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            if res == "Original":
                return image
            else:
                vlm_size = VLM_RESOLUTIONS[res]
                total_vlm = vlm_size * vlm_size
                scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                width_vlm = round(samples.shape[3] * scale_by_vlm)
                height_vlm = round(samples.shape[2] * scale_by_vlm)

                s_vlm = common_upscale(samples, width_vlm, height_vlm, "bicubic", "disabled")
                return s_vlm.movedim(1, -1)

        # Create dict of preprocessed VLM images for math evaluation
        processed_images = {}
        for num, img in raw_images.items():
            name = ImageInputMapping.get_display_name(num, is_zero_indexed)
            processed_images[name] = process_vlm_image(img, vlm_resolution)

        # Parse and replace any math formulas enclosed in pipes |formula|
        math_pattern = re.compile(r"\|([^|]+)\|")

        def replace_formula(match):
            expression = match.group(1).strip()
            result_tensor = evaluate_formula(expression, processed_images)
            images_vl.append(result_tensor)
            return "<|vision_start|><|image_pad|><|vision_end|>"

        modified_prompt = math_pattern.sub(replace_formula, prompt)

        # Re-check for keywords in the modified prompt
        has_keywords = bool(pattern.search(modified_prompt)) or len(images_vl) > 0

        if has_keywords:
            # Replace keywords dynamically and build images_vl in order of appearance
            def replace_keyword(match):
                num = int(match.group(1))
                dict_key = ImageInputMapping.get_dict_key(num, is_zero_indexed)
                if dict_key in raw_images:
                    img = raw_images[dict_key]
                    processed_img = processed_images.get(ImageInputMapping.get_display_name(dict_key, is_zero_indexed), process_vlm_image(img, vlm_resolution))
                    images_vl.append(processed_img)
                    return "<|vision_start|><|image_pad|><|vision_end|>"
                return ""

            modified_prompt = pattern.sub(replace_keyword, modified_prompt)
        else:
            # Fallback: prepend all connected images in numerical order of their slots
            image_prompt = ""
            for num in sorted(raw_images.keys()):
                name = ImageInputMapping.get_display_name(num, is_zero_indexed)
                processed_img = processed_images[name]
                images_vl.append(processed_img)
                image_prompt += f"<|vision_start|><|image_pad|><|vision_end|>"

            modified_prompt = image_prompt + modified_prompt

        # Pass standard tokens to tokenize (with images mapped to tags) and encode
        tokens = clip.tokenize(modified_prompt, images=images_vl)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class TextEncodeGemmaSystemEditPlusAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="TextEncodeGemmaSystemEditPlusAdvanced",
            display_name="TextEncodeGemmaSystemEditPlusAdvanced",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path). 'Fast' = 384x384, 'Balanced' = 512x512, 'Detailed' = 768x768, 'Large' = 1024x1024, 'X-Large' = 1280x1280, 'XX-Large' = 1536x1536, 'Original' uses native resolution.",
                ),
                io.Autogrow.Input("image_inputs", template=autogrow_template),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt, vlm_resolution, image_inputs: io.Autogrow.Type) -> io.NodeOutput:
        # Collect and parse all autogrow keys
        raw_images = {}
        if image_inputs is not None:
            for k, v in image_inputs.items():
                if v is not None:
                    # Extract numeric suffix (e.g. "image1" -> 1)
                    digits = re.findall(r'\d+', k)
                    if digits:
                        idx = int(digits[0])
                    else:
                        idx = 1
                    raw_images[idx] = v

        # Determine indexing: 0-indexed or 1-indexed.
        is_zero_indexed = 0 in raw_images

        # Check if the prompt has any image_input_ keyword matches (case-insensitive)
        pattern = re.compile(r'image_input_(\d+)', re.IGNORECASE)
        has_keywords = bool(pattern.search(prompt))

        images_vl_raw = []

        if has_keywords:
            # Replace keywords dynamically and build images_vl_raw in order of appearance
            def replace_keyword(match):
                num = int(match.group(1))
                dict_key = num - 1 if is_zero_indexed else num
                if dict_key in raw_images:
                    img = raw_images[dict_key]
                    images_vl_raw.append(img)
                    return "<img><image_soft_token><end_of_image>"
                return ""

            modified_prompt = pattern.sub(replace_keyword, prompt)
        else:
            # Fallback: prepend all connected images in numerical order of their slots
            image_prompt = ""
            for num in sorted(raw_images.keys()):
                img = raw_images[num]
                images_vl_raw.append(img)
                display_num = num + 1 if is_zero_indexed else num
                image_prompt += f"<img><image_soft_token><end_of_image>"

            modified_prompt = image_prompt + prompt

        # Construct the complete template string via safe concatenation
        if len(system_prompt) > 0:
            full_prompt = (
                "<start_of_turn>system\n" + system_prompt + "<end_of_turn>\n" +
                "<start_of_turn>user\n" + modified_prompt + "<end_of_turn>\n<start_of_turn>model\n"
            )
        else:
            full_prompt = (
                "<start_of_turn>system\nYou are a helpful assistant.<end_of_turn>\n" +
                "<start_of_turn>user\n" + modified_prompt + "<end_of_turn>\n<start_of_turn>model\n"
            )

        # 1. First tokenize the text without passing images, getting raw 262144 token IDs
        tokens = clip.tokenize(full_prompt, skip_template=True)

        # 2. Helper to process image for VLM
        def process_vlm_image(image, res):
            if image is None:
                return None
            if res == "Original":
                return image[:, :, :, :3]
            VLM_RESOLUTIONS = {
                "Fast (384)": 384,
                "Balanced (512)": 512,
                "Detailed (768)": 768,
                "Large (1024)": 1024,
                "X-Large (1280)": 1280,
                "XX-Large (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            vlm_size = VLM_RESOLUTIONS[res]
            total_vlm = vlm_size * vlm_size
            scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
            width_vlm = round(samples.shape[3] * scale_by_vlm)
            height_vlm = round(samples.shape[2] * scale_by_vlm)

            s_vlm = common_upscale(samples, width_vlm, height_vlm, "bicubic", "disabled")
            return s_vlm.movedim(1, -1)[:, :, :, :3]

        # 3. Process the images and manually inject them sequentially into the 262144 tokens
        if len(images_vl_raw) > 0:
            processed_images = [process_vlm_image(img, vlm_resolution) for img in images_vl_raw]

            # Loop over all tokenizer sections (e.g. 'gemma3_12b')
            for key, val in tokens.items():
                if isinstance(val, list):
                    embed_count = 0
                    for r in val:
                        if isinstance(r, list):
                            for i, token in enumerate(r):
                                if isinstance(token, tuple) and len(token) > 0:
                                    if token[0] == 262144 and embed_count < len(processed_images):
                                        # Replace the token ID (index 0 of the tuple) with the visual payload dict
                                        r[i] = ({"type": "image", "data": processed_images[embed_count]},) + token[1:]
                                        embed_count += 1

        # 4. Encode from the modified tokens dict
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class UC_TextEncodeLtxv2SystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeLtxv2SystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with LTXV 2 System Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Image.Input("image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt="", image=None) -> io.NodeOutput:
        # Build template with string concat (ComfyUI pattern)
        if image is not None:
            llama_template = (
                "<start_of_turn>system\n" + system_prompt + "<end_of_turn>\n" +
                "<start_of_turn>user\n\n<image_soft_token>{}<end_of_turn>\n\n<start_of_turn>model\n"
            )
        elif len(system_prompt) > 0:
            llama_template = (
                "<start_of_turn>system\n" + system_prompt + "<end_of_turn>\n" +
                "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
            )
        else:
            llama_template = (
                "<start_of_turn>system\nYou are a helpful assistant.<end_of_turn>\n<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
            )

        if image is not None:
            tokens = clip.tokenize(prompt, llama_template=llama_template, image=image)
        else:
            tokens = clip.tokenize(prompt, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class UC_TextEncodeZITSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeZITSystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Z-Image System Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt=None) -> io.NodeOutput:
        if len(system_prompt) > 0:
            template_prefix = "<|im_start|>system\n"
            template_suffix = (
                "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            )
            llama_template = f"{template_prefix}{system_prompt}{template_suffix}"
            tokens = clip.tokenize(prompt, llama_template=llama_template)
        else:
            tokens = clip.tokenize(prompt)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class UC_TextEncodeZImageThinkPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeZImageThinkPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Z-Image Thinking Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("thinking", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, thinking=None) -> io.NodeOutput:
        if len(thinking) > 0:
            template_prefix = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            template_suffix = "\n</think>\n\n"
            llama_template = f"{template_prefix}{thinking}{template_suffix}"
            tokens = clip.tokenize(prompt, llama_template=llama_template)
        else:
            tokens = clip.tokenize(prompt)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


# Template definitions for unified node
SYSTEM_PROMPT_TEMPLATES = {
    "flux2dev": {
        "prefix": r"[SYSTEM_PROMPT]",
        "suffix": r"[/SYSTEM_PROMPT][INST]{}[/INST]",
    },
    "klein": {
        "prefix": "<|im_start|>system\n",
        "suffix": "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    },
    "z-image": {
        "prefix": "<|im_start|>system\n",
        "suffix": "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    },
}


class UC_TextEncodeSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeSystemPrompt",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input(
                    "model_type",
                    options=["flux2dev", "klein", "z-image"],
                    default="flux2dev",
                    tooltip="Select the model type to use the correct template format.",
                ),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.String.Input(
                    "thinking_content",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip="(Klein only) Custom thinking content to inject. Leave empty for default.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, model_type, prompt, system_prompt="", thinking_content="") -> io.NodeOutput:
        if model_type == "klein" and len(thinking_content) > 0:
            # Klein with custom thinking content
            if len(system_prompt) > 0:
                llama_template = (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n" +
                    f"<|im_start|>user\n{{}}<|im_end|>\n" +
                    f"<|im_start|>assistant\n<think>\n{thinking_content}\n</think>\n\n"
                )
            else:
                llama_template = (
                    "<|im_start|>user\n{}<|im_end|>\n" +
                    f"<|im_start|>assistant\n<think>\n{thinking_content}\n</think>\n\n"
                )
            tokens = clip.tokenize(prompt, llama_template=llama_template)
        elif len(system_prompt) > 0:
            template = SYSTEM_PROMPT_TEMPLATES.get(model_type, SYSTEM_PROMPT_TEMPLATES["flux2dev"])
            llama_template = f"{template['prefix']}{system_prompt}{template['suffix']}"
            tokens = clip.tokenize(prompt, llama_template=llama_template)
        else:
            tokens = clip.tokenize(prompt)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class UC_TextGenerateQwen35SystemPrompt(io.ComfyNode):
    """
    TextGenerate variant for Qwen3.5 models with custom system message support.
    Builds the chat template via string concatenation (never .format()) so any
    characters in user input — including { } \\ and control sequences — are safe.
    """

    @classmethod
    def define_schema(cls):
        sampling_options = [
            io.DynamicCombo.Option(
                key="on",
                inputs=[
                    io.Float.Input("temperature", default=0.7, min=0.01, max=2.0, step=0.000001),
                    io.Int.Input("top_k", default=64, min=0, max=1000),
                    io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("min_p", default=0.05, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("repetition_penalty", default=1.05, min=0.0, max=5.0, step=0.01),
                    io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                    io.Float.Input("presence_penalty", optional=True, default=0.0, min=0.0, max=5.0, step=0.01),
                ]
            ),
            io.DynamicCombo.Option(
                key="off",
                inputs=[]
            ),
        ]

        return io.Schema(
            node_id="UC_TextGenerateQwen35SystemPrompt",
            display_name="Text Generate Qwen3.5 (System Prompt)",
            category="advanced/textgen",
            search_aliases=["LLM", "VLM", "qwen", "qwen35", "system prompt", "textgen"],
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=False, default="",
                    tooltip="User message. All characters including { } are safe — template uses concatenation not .format()."),
                io.String.Input("system_message", multiline=True, dynamic_prompts=False, default="",
                    tooltip="System message injected before the user turn. Leave empty to skip the system block entirely."),
                io.Image.Input("image", optional=True),
                io.Int.Input("max_length", default=512, min=1, max=8192),
                io.DynamicCombo.Input("sampling_mode", options=sampling_options, display_name="Sampling Mode"),
                io.Boolean.Input("thinking", optional=True, default=False,
                    tooltip="Enable thinking mode. When False, suppresses thinking with <think>\\n</think>\\n."),
            ],
            outputs=[
                io.String.Output(display_name="generated_text"),
            ],
        )

    @classmethod
    def _build_prompt(cls, prompt: str, system_message: str, has_image: bool, thinking: bool) -> str:
        """
        Build the full Qwen3.5 chat-template string using concatenation only.
        No .format() / %-formatting — user-supplied strings are never interpolated,
        so { } and any other characters are completely safe.

        Template structure (matches qwen35.py Qwen35ImageTokenizer):
            [<|im_start|>system\\n{system_message}<|im_end|>\\n]   <- optional
            <|im_start|>user\\n
            [<|vision_start|><|image_pad|><|vision_end|>]          <- if image
            {prompt}<|im_end|>\\n
            <|im_start|>assistant\\n
            [<think>\\n</think>\\n]                                  <- if not thinking
        """
        result = ""

        # Optional system block
        if system_message and system_message.strip():
            result = (
                "<|im_start|>system\n"
                + system_message
                + "<|im_end|>\n"
            )

        # User block — image token placed before text per qwen35.py:761
        result += "<|im_start|>user\n"
        if has_image:
            result += "<|vision_start|><|image_pad|><|vision_end|>"
        result += prompt + "<|im_end|>\n"

        # Assistant block
        result += "<|im_start|>assistant\n"

        # Suppress thinking unless thinking mode requested (matches qwen35.py:784-785)
        if not thinking:
            result += "<think>\n</think>\n"

        return result

    @classmethod
    def execute(cls, clip, prompt, system_message, max_length, sampling_mode,
                image=None, thinking=False) -> io.NodeOutput:

        formatted_prompt = cls._build_prompt(
            prompt=prompt,
            system_message=system_message,
            has_image=image is not None,
            thinking=thinking,
        )

        # skip_template=True because we built the full template ourselves.
        # The tokenizer detects <|im_start|> prefix and skips its own template (qwen35.py:769).
        tokens = clip.tokenize(
            formatted_prompt,
            image=image,
            skip_template=True,
            min_length=1,
            thinking=thinking,
        )

        do_sample = sampling_mode.get("sampling_mode") == "on"
        temperature = sampling_mode.get("temperature", 1.0)
        top_k = sampling_mode.get("top_k", 50)
        top_p = sampling_mode.get("top_p", 1.0)
        min_p = sampling_mode.get("min_p", 0.0)
        seed = sampling_mode.get("seed", None)
        repetition_penalty = sampling_mode.get("repetition_penalty", 1.0)
        presence_penalty = sampling_mode.get("presence_penalty", 0.0)

        generated_ids = clip.generate(
            tokens,
            do_sample=do_sample,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
        )

        generated_text = clip.decode(generated_ids, skip_special_tokens=True)
        return io.NodeOutput(generated_text)


class TextEncodeKrea2SystemEditScaledAdv(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="TextEncodeKrea2SystemEditScaledAdv",
            display_name="TextEncodeKrea2SystemEditScaledAdv",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="Main text prompt. Supports classical weight syntax: (prompt:weight), e.g. (sunset:1.2).",
                ),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path). 'Fast' = 384x384, 'Balanced' = 512x512, 'Detailed' = 768x768, 'Large' = 1024x1024, 'X-Large' = 1280x1280, 'XX-Large' = 1536x1536, 'Original' uses native resolution.",
                ),
                io.Float.Input("multiplier", default=1.0, min=-1000.0, max=1000.0, step=0.1, tooltip="Overall multiplier applied to the final conditioning vector."),
                io.Combo.Input(
                    "padding_method",
                    options=["zero-pad", "interpolate"],
                    default="zero-pad",
                    tooltip="Alignment method for images with different aspect ratios/resolutions. 'zero-pad' pads with zeros (matches ComfyUI core), 'interpolate' linearly resizes visual tokens to preserve attention alignment.",
                ),
                io.String.Input(
                    "formula",
                    default="a",
                    multiline=False,
                    tooltip="Mathematical formula to blend conditioning outputs. Use variables a, b, c, d... to reference active connected image inputs. Supports weight syntax inside the formula, e.g. (a:1.2) + (b:0.8)",
                ),
                io.Autogrow.Input("image_inputs", template=autogrow_template),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt, vlm_resolution, multiplier, padding_method, formula, image_inputs: io.Autogrow.Type) -> io.NodeOutput:
        # Collect and parse all active (non-null) connected images sequentially
        active_images = []
        if image_inputs is not None:
            for k, v in image_inputs.items():
                if v is not None:
                    active_images.append(v)

        if not active_images:
            # Fallback if no images are connected: encode prompt as plain text
            conditioning = encode_embedding_classical_scaled_bias(clip, prompt, multiplier=multiplier)
            return io.NodeOutput(conditioning)

        # Map active images sequentially to letter variables (a, b, c, ...) and encode each pass
        sequence_tensors = {}
        pooled_tensors = {}
        deepstack_dict = {}
        last_cond_dict = None

        def process_vlm_image(image, res):
            if image is None:
                return None
            VLM_RESOLUTIONS = {
                "Fast (384)": 384,
                "Balanced (512)": 512,
                "Detailed (768)": 768,
                "Large (1024)": 1024,
                "X-Large (1280)": 1280,
                "XX-Large (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            if res == "Original":
                return image
            else:
                vlm_size = VLM_RESOLUTIONS[res]
                total_vlm = vlm_size * vlm_size
                scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                width_vlm = round(samples.shape[3] * scale_by_vlm)
                height_vlm = round(samples.shape[2] * scale_by_vlm)

                s_vlm = common_upscale(samples, width_vlm, height_vlm, "bicubic", "disabled")
                return s_vlm.movedim(1, -1)

        for idx, img in enumerate(active_images):
            letter = chr(97 + idx)  # 0 -> 'a', 1 -> 'b', 2 -> 'c', ...
            processed_img = process_vlm_image(img, vlm_resolution)

            # Ensure prompt has image pad tokens so tokenizer knows where to inject the image
            modified_prompt = prompt
            if not any(tag in prompt for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>", "image_input_"]):
                modified_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + modified_prompt

            # Wrap in Llama/Krea2 system prompt template format
            if len(system_prompt) > 0:
                full_prompt = (
                    "<|im_start|>user\n" + "<|im_end|>\n" +
                    "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                    "<|im_start|>user\n" + modified_prompt + "<|im_end|>\n" +
                    "<|im_start|>assistant\n"
                )
            else:
                full_prompt = (
                    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
                    "<|im_start|>user\n" + modified_prompt + "<|im_end|>\n" +
                    "<|im_start|>assistant\n"
                )

            # Encode individual sequence pass
            cond_X = encode_embedding_classical_scaled_bias(clip, full_prompt, images=[processed_img], skip_template=True)
            C_X = cond_X[0][0]
            P_X = cond_X[0][1].get("pooled_output", None)

            sequence_tensors[letter] = C_X
            if P_X is not None:
                pooled_tensors[letter] = P_X

            # Extract DeepStack per-layer tensors if present
            if "embeds_info" in cond_X[0][1] and len(cond_X[0][1]["embeds_info"]) > 0:
                extra = cond_X[0][1]["embeds_info"][0].get("extra", {})
                if "deepstack" in extra:
                    deepstack_dict[letter] = extra["deepstack"]

            last_cond_dict = cond_X[0][1]

        # Evaluate mathematical formula on sequence and pooled tensors
        C_blended, P_blended = evaluate_conditioning_formula(formula, sequence_tensors, pooled_tensors, padding_method=padding_method)

        # Evaluate mathematical formula on DeepStack layers
        deepstack_blended = None
        if deepstack_dict:
            first_key = next(iter(sequence_tensors.keys()))
            num_layers = len(deepstack_dict[first_key])
            max_vis_len = max(ds_list[0].shape[0] for ds_list in deepstack_dict.values())

            deepstack_blended = []
            import torch.nn.functional as F

            for l in range(num_layers):
                layer_tensors = {let: ds_list[l] for let, ds_list in deepstack_dict.items()}

                # Align layer tensors to maximum length
                aligned_layer_tensors = {}
                for name, tensor in layer_tensors.items():
                    if tensor.shape[0] < max_vis_len:
                        if padding_method == "interpolate":
                            tensor_perm = tensor.permute(1, 0).unsqueeze(0)
                            tensor_interp = F.interpolate(tensor_perm, size=max_vis_len, mode='linear', align_corners=False)
                            tensor = tensor_interp.squeeze(0).permute(1, 0)
                        else:  # zero-pad
                            pad_size = max_vis_len - tensor.shape[0]
                            padding = torch.zeros((pad_size, tensor.shape[1]), device=tensor.device, dtype=tensor.dtype)
                            tensor = torch.cat([tensor, padding], dim=0)
                    aligned_layer_tensors[name] = tensor

                safe_dict_layer = {
                    "__builtins__": {},
                    "clamp": torch.clamp,
                    "min": torch.minimum,
                    "max": torch.maximum,
                    "abs": torch.abs,
                }
                for name, t in aligned_layer_tensors.items():
                    safe_dict_layer[name] = t

                # Handle classical weighting conversions inside math formula
                layer_expression = re.sub(
                    r"\(\s*([a-zA-Z0-9_]+)\s*:\s*([0-9.-]+)\s*\)",
                    r"(\1 * \2)",
                    formula
                )

                try:
                    layer_blended = eval(layer_expression, safe_dict_layer, {}) # noqa: S307
                    deepstack_blended.append(layer_blended)
                except Exception as e:
                    raise RuntimeError(f"Error evaluating DeepStack math expression at layer {l}: {e}")

        # Build final conditioning dictionary
        final_cond_dict = last_cond_dict.copy()
        if P_blended is not None:
            final_cond_dict["pooled_output"] = P_blended

        if deepstack_blended is not None and "embeds_info" in final_cond_dict and len(final_cond_dict["embeds_info"]) > 0:
            final_cond_dict["embeds_info"] = [final_cond_dict["embeds_info"][0].copy()]
            final_cond_dict["embeds_info"][0]["extra"] = final_cond_dict["embeds_info"][0]["extra"].copy()
            final_cond_dict["embeds_info"][0]["extra"]["deepstack"] = deepstack_blended

        if multiplier != 1.0:
            C_blended *= multiplier
            if "pooled_output" in final_cond_dict and final_cond_dict["pooled_output"] is not None:
                final_cond_dict["pooled_output"] *= multiplier

        return io.NodeOutput([[C_blended, final_cond_dict]])


class TextEncodeEditScaledAdv(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="TextEncodeEditScaledAdv",
            display_name="TextEncodeEditScaledAdv",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="Main text prompt. Supports classical weight syntax: (prompt:weight), e.g. (sunset:1.2).",
                ),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path). 'Fast' = 384x384, 'Balanced' = 512x512, 'Detailed' = 768x768, 'Large' = 1024x1024, 'X-Large' = 1280x1280, 'XX-Large' = 1536x1536, 'Original' uses native resolution.",
                ),
                io.Float.Input("multiplier", default=1.0, min=-1000.0, max=1000.0, step=0.1, tooltip="Overall multiplier applied to the final conditioning vector."),
                io.Combo.Input(
                    "padding_method",
                    options=["zero-pad", "interpolate"],
                    default="zero-pad",
                    tooltip="Alignment method for images with different aspect ratios/resolutions. 'zero-pad' pads with zeros (matches ComfyUI core), 'interpolate' linearly resizes visual tokens to preserve attention alignment.",
                ),
                io.String.Input(
                    "formula",
                    default="a",
                    multiline=False,
                    tooltip="Mathematical formula to blend conditioning outputs. Use variables a, b, c, d... to reference active connected image inputs. Supports weight syntax inside the formula, e.g. (a:1.2) + (b:0.8)",
                ),
                io.Autogrow.Input("image_inputs", template=autogrow_template),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vlm_resolution, multiplier, padding_method, formula, image_inputs: io.Autogrow.Type) -> io.NodeOutput:
        # Collect and parse all active (non-null) connected images sequentially
        active_images = []
        if image_inputs is not None:
            for k, v in image_inputs.items():
                if v is not None:
                    active_images.append(v)

        if not active_images:
            # Fallback if no images are connected: encode prompt as plain text
            conditioning = encode_embedding_classical_scaled_bias(clip, prompt, multiplier=multiplier)
            return io.NodeOutput(conditioning)

        # Map active images sequentially to letter variables (a, b, c, ...) and encode each pass
        sequence_tensors = {}
        pooled_tensors = {}
        last_cond_dict = None

        def process_vlm_image(image, res):
            if image is None:
                return None
            VLM_RESOLUTIONS = {
                "Fast (384)": 384,
                "Balanced (512)": 512,
                "Detailed (768)": 768,
                "Large (1024)": 1024,
                "X-Large (1280)": 1280,
                "XX-Large (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            if res == "Original":
                return image
            else:
                vlm_size = VLM_RESOLUTIONS[res]
                total_vlm = vlm_size * vlm_size
                scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                width_vlm = round(samples.shape[3] * scale_by_vlm)
                height_vlm = round(samples.shape[2] * scale_by_vlm)

                s_vlm = common_upscale(samples, width_vlm, height_vlm, "bicubic", "disabled")
                return s_vlm.movedim(1, -1)

        for idx, img in enumerate(active_images):
            letter = chr(97 + idx)  # 0 -> 'a', 1 -> 'b', 2 -> 'c', ...
            processed_img = process_vlm_image(img, vlm_resolution)

            # Ensure prompt has image pad tokens so tokenizer knows where to inject the image
            modified_prompt = prompt
            if not any(tag in prompt for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>", "image_input_"]):
                modified_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + modified_prompt

            # Encode individual sequence pass
            cond_X = encode_embedding_classical_scaled_bias(clip, modified_prompt, images=[processed_img])
            C_X = cond_X[0][0]
            P_X = cond_X[0][1].get("pooled_output", None)

            sequence_tensors[letter] = C_X
            if P_X is not None:
                pooled_tensors[letter] = P_X
            last_cond_dict = cond_X[0][1]

        # Evaluate mathematical formula on sequence and pooled tensors
        C_blended, P_blended = evaluate_conditioning_formula(formula, sequence_tensors, pooled_tensors, padding_method=padding_method)

        # Build final conditioning dictionary
        final_cond_dict = last_cond_dict.copy()
        if P_blended is not None:
            final_cond_dict["pooled_output"] = P_blended

        if multiplier != 1.0:
            C_blended *= multiplier
            if "pooled_output" in final_cond_dict and final_cond_dict["pooled_output"] is not None:
                final_cond_dict["pooled_output"] *= multiplier

        return io.NodeOutput([[C_blended, final_cond_dict]])


class UC_Krea2InputEmbeds(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_Krea2InputEmbeds",
            display_name="Krea 2 Input Embeddings",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, default="", tooltip="Input text prompt. Important: skips any template wrapping."),
                io.Image.Input("image", optional=True, tooltip="Optional image input to interleave."),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path).",
                ),
                io.String.Input("file_name", default="qwen3vl_4b_embed", tooltip="Specify the file name. The embedding will be saved as {file_name}.safetensors directly inside your ComfyUI embeddings directory."),
                io.Boolean.Input(
                    "slice_visual_tokens",
                    default=False,
                    tooltip="If True, performs perfect visual slicing (Method A) to cut out visual tokens, saving a pure language embedding. If False (default), preserves the full interleaved sequence including visual tokens.",
                ),
            ],
            outputs=[
                io.AnyType.Output("state_dict", tooltip="Dictionary structure: {'qwen3vl_4b': tensor_2d} of shape [num_tokens, 2560]"),
                io.AnyType.Output("tensor_2d", tooltip="Raw PyTorch 2D tensor of shape [num_tokens, 2560]"),
            ]
        )

    @classmethod
    def execute(cls, clip, prompt, vlm_resolution, image=None, file_name="qwen3vl_4b_embed", slice_visual_tokens=False) -> io.NodeOutput:
        # Preprocess image if present
        processed_img = None
        images_vl = []
        if image is not None:
            def process_vlm_image(img, res):
                VLM_RESOLUTIONS = {
                    "Fast (384)": 384,
                    "Balanced (512)": 512,
                    "Detailed (768)": 768,
                    "Large (1024)": 1024,
                    "X-Large (1280)": 1280,
                    "XX-Large (1536)": 1536
                }
                samples = img.movedim(-1, 1)
                if res == "Original":
                    return img
                else:
                    vlm_size = VLM_RESOLUTIONS[res]
                    total_vlm = vlm_size * vlm_size
                    scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                    width_vlm = round(samples.shape[3] * scale_by_vlm)
                    height_vlm = round(samples.shape[2] * scale_by_vlm)

                    s_vlm = common_upscale(samples, width_vlm, height_vlm, "bicubic", "disabled")
                    return s_vlm.movedim(1, -1)
            processed_img = process_vlm_image(image, vlm_resolution)
            images_vl.append(processed_img)

        # Tokenize prompt using skip_template=True so no template wrapping is saved
        modified_prompt = prompt
        if image is not None:
            if not any(tag in prompt for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>"]):
                modified_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + modified_prompt

        tokens = clip.tokenize(modified_prompt, images=images_vl, skip_template=True)

        # Retrieve the key name (typically "qwen3vl_4b")
        key_name = next(iter(tokens.keys()))
        token_list = tokens[key_name]
        tokens_only = [[t[0] for t in b] for b in token_list]

        # Process the tokens into raw interleaved input embeddings
        cond_stage = clip.cond_stage_model
        clip_model = None

        if hasattr(cond_stage, "clip") and isinstance(cond_stage.clip, str) and hasattr(cond_stage, cond_stage.clip):
            clip_model = getattr(cond_stage, cond_stage.clip)
        elif hasattr(cond_stage, "clip_model"):
            clip_model = cond_stage.clip_model
        elif hasattr(cond_stage, "clip_d"):
            clip_model = cond_stage.clip_d
        else:
            clip_model = cond_stage

        if clip_model is None or not hasattr(clip_model, "process_tokens"):
            raise AttributeError("Could not locate underlying model wrapper with 'process_tokens' method in cond_stage_model.")

        embeds, _, _, embeds_info = clip_model.process_tokens(tokens_only, clip_model.execution_device)

        # Locate and slice out any visual tokens to save ONLY the pure language/text tokens
        if slice_visual_tokens:
            vis_start, vis_end = find_visual_token_range(tokens, embeds)
            if vis_start < vis_end:
                prefix = embeds[:, :vis_start, :]
                suffix = embeds[:, vis_end:, :]
                embeds_sliced = torch.cat([prefix, suffix], dim=1)
            else:
                embeds_sliced = embeds
        else:
            embeds_sliced = embeds

        # Convert to 2D tensor [num_tokens, 2560]
        # embeds_sliced is shape (1, num_tokens, 2560)
        tensor_2d = embeds_sliced.squeeze(0).clone().cpu()

        # Build state dict
        state_dict = {"qwen3vl_4b": tensor_2d}

        # If file_name is provided, save it as a .safetensors file in ComfyUI's embeddings directory
        if file_name and file_name.strip():
            import folder_paths
            from safetensors.torch import save_file

            # Locate ComfyUI embeddings directory
            try:
                embed_paths = folder_paths.get_folder_paths("embeddings")
                if embed_paths:
                    embeddings_dir = embed_paths[0]
                else:
                    embeddings_dir = os.path.join(os.path.dirname(folder_paths.__file__), "models", "embeddings")
            except Exception:
                embeddings_dir = "models/embeddings"

            os.makedirs(embeddings_dir, exist_ok=True)
            target_path = os.path.join(embeddings_dir, f"{file_name.strip()}.safetensors")

            # Save the state dict using safetensors (tensors must be contiguous)
            state_dict_safe = {k: v.contiguous() for k, v in state_dict.items()}
            save_file(state_dict_safe, target_path)

        return io.NodeOutput(state_dict, tensor_2d)


class UC_Qwen3VLInputEmbeds(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_Qwen3VLInputEmbeds",
            display_name="Qwen3-VL Unified Input Embeddings",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, default="", tooltip="Input text prompt. Important: skips any template wrapping."),
                io.Image.Input("image", optional=True, tooltip="Optional image input to interleave."),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path).",
                ),
                io.String.Input("file_name", default="qwen3vl_embed", tooltip="Specify the file name. The embedding will be saved as {file_name}.safetensors directly inside your ComfyUI embeddings directory."),
                io.Boolean.Input(
                    "slice_visual_tokens",
                    default=False,
                    tooltip="If True, performs perfect visual slicing (Method A) to cut out visual tokens, saving a pure language embedding. If False (default), preserves the full interleaved sequence including visual tokens.",
                ),
            ],
            outputs=[
                io.AnyType.Output("state_dict", tooltip="Dictionary structure: {key_name: tensor_2d} of shape [num_tokens, hidden_size]"),
                io.AnyType.Output("tensor_2d", tooltip="Raw PyTorch 2D tensor of shape [num_tokens, hidden_size]"),
            ]
        )

    @classmethod
    def execute(cls, clip, prompt, vlm_resolution, image=None, file_name="qwen3vl_embed", slice_visual_tokens=False) -> io.NodeOutput:
        # Preprocess image if present
        processed_img = None
        images_vl = []
        if image is not None:
            def process_vlm_image(img, res):
                VLM_RESOLUTIONS = {
                    "Fast (384)": 384,
                    "Balanced (512)": 512,
                    "Detailed (768)": 768,
                    "Large (1024)": 1024,
                    "X-Large (1280)": 1280,
                    "XX-Large (1536)": 1536
                }
                samples = img.movedim(-1, 1)
                if res == "Original":
                    return img
                else:
                    vlm_size = VLM_RESOLUTIONS[res]
                    total_vlm = vlm_size * vlm_size
                    scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                    width_vlm = round(samples.shape[3] * scale_by_vlm)
                    height_vlm = round(samples.shape[2] * scale_by_vlm)

                    s_vlm = common_upscale(samples, width_vlm, height_vlm, "bicubic", "disabled")
                    return s_vlm.movedim(1, -1)
            processed_img = process_vlm_image(image, vlm_resolution)
            images_vl.append(processed_img)

        # Tokenize prompt using skip_template=True so no template wrapping is saved
        modified_prompt = prompt
        if image is not None:
            if not any(tag in prompt for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>"]):
                modified_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + modified_prompt

        tokens = clip.tokenize(modified_prompt, images=images_vl, skip_template=True)

        # Retrieve the key name dynamically (typically "qwen3vl_4b" or "qwen3vl_8b")
        key_name = "qwen3vl_8b"
        if tokens:
            key_name = next(iter(tokens.keys()))
        token_list = tokens.get(key_name, [])
        tokens_only = [[t[0] for t in b] for b in token_list]

        # Process the tokens into raw interleaved input embeddings
        cond_stage = clip.cond_stage_model
        clip_model = None

        if hasattr(cond_stage, "clip") and isinstance(cond_stage.clip, str) and hasattr(cond_stage, cond_stage.clip):
            clip_model = getattr(cond_stage, cond_stage.clip)
        elif hasattr(cond_stage, "clip_model"):
            clip_model = cond_stage.clip_model
        elif hasattr(cond_stage, "clip_d"):
            clip_model = cond_stage.clip_d
        else:
            clip_model = cond_stage

        if clip_model is None or not hasattr(clip_model, "process_tokens"):
            raise AttributeError("Could not locate underlying model wrapper with 'process_tokens' method in cond_stage_model.")

        embeds, _, _, embeds_info = clip_model.process_tokens(tokens_only, clip_model.execution_device)

        # Locate and slice out any visual tokens to save ONLY the pure language/text tokens
        if slice_visual_tokens:
            vis_start, vis_end = find_visual_token_range(tokens, embeds)
            if vis_start < vis_end:
                prefix = embeds[:, :vis_start, :]
                suffix = embeds[:, vis_end:, :]
                embeds_sliced = torch.cat([prefix, suffix], dim=1)
            else:
                embeds_sliced = embeds
        else:
            embeds_sliced = embeds

        # Convert to 2D tensor [num_tokens, hidden_size]
        tensor_2d = embeds_sliced.squeeze(0).clone().cpu()

        # Build state dict
        state_dict = {key_name: tensor_2d}

        # If file_name is provided, save it as a .safetensors file in ComfyUI's embeddings directory
        if file_name and file_name.strip():
            import folder_paths
            from safetensors.torch import save_file

            # Locate ComfyUI embeddings directory
            try:
                embed_paths = folder_paths.get_folder_paths("embeddings")
                if embed_paths:
                    embeddings_dir = embed_paths[0]
                else:
                    embeddings_dir = os.path.join(os.path.dirname(folder_paths.__file__), "models", "embeddings")
            except Exception:
                embeddings_dir = "models/embeddings"

            os.makedirs(embeddings_dir, exist_ok=True)
            target_path = os.path.join(embeddings_dir, f"{file_name.strip()}.safetensors")

            # Save the state dict using safetensors (tensors must be contiguous)
            state_dict_safe = {k: v.contiguous() for k, v in state_dict.items()}
            save_file(state_dict_safe, target_path)

        return io.NodeOutput(state_dict, tensor_2d)


_QWEN_IM_START, _QWEN_USER, _QWEN_NL, _QWEN_IM_END = 151644, 872, 198, 151645

def _krea2_user_content_span(ids):
    best_start, best_end = None, None
    for i in range(len(ids) - 2):
        if ids[i] == _QWEN_IM_START and ids[i + 1] == _QWEN_USER and ids[i + 2] == _QWEN_NL:
            start = i + 3
            end = start
            while end < len(ids) and ids[end] != _QWEN_IM_END:
                end += 1
            if end > start:  # prioritize non-empty block
                best_start, best_end = start, end
    if best_start is not None:
        return best_start, best_end
    for i in range(len(ids) - 2):
        if ids[i] == _QWEN_IM_START and ids[i + 1] == _QWEN_USER and ids[i + 2] == _QWEN_NL:
            start = i + 3
            end = start
            while end < len(ids) and ids[end] != _QWEN_IM_END:
                end += 1
            return start, end
    return None, None

def _krea2_token_ids(clip, text):
    tok = clip.tokenize(text)
    key = next(iter(tok))
    return [t[0] if isinstance(t, tuple) else t for t in tok[key][0]]

def _find_subsequence(seq, sub, lo, hi):
    out = []
    n = len(sub)
    if n == 0:
        return out
    for i in range(lo, hi - n + 1):
        if seq[i:i + n] == sub:
            out.append(i)
    return out


def krea2_attn_forward_weight(self, x, freqs=None, mask=None, transformer_options={}):
    from einops import rearrange
    from comfy.ldm.flux.math import apply_rope
    from comfy.ldm.modules.attention import optimized_attention, attention_pytorch

    q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
    q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
    k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
    v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)

    weights = transformer_options.get("krea2_token_weights")
    if weights:
        v = v.clone()
        for pos, v_factor, _ in weights:
            if v_factor != 1.0 and pos < v.shape[2]:
                v[:, :, pos] = v[:, :, pos] * v_factor
    q, k = self.qknorm(q, k)
    if freqs is not None:
        q, k = apply_rope(q, k, freqs)
    if self.kvheads != self.heads:
        rep = self.heads // self.kvheads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    bias = None
    if weights and any(kb != 0.0 for _, _, kb in weights):
        bias = q.new_zeros(1, k.shape[2])
        for pos, _, kb in weights:
            if kb != 0.0 and pos < bias.shape[1]:
                bias[:, pos] = kb
    if bias is not None:
        out = attention_pytorch(q, k, v, self.heads, mask=bias, skip_reshape=True)
    else:
        out = optimized_attention(q, k, v, self.heads, mask=mask, skip_reshape=True, transformer_options=transformer_options)
    return self.wo(out * torch.sigmoid(gate))

class Krea2WeightPatch:
    def __get__(self, obj, objtype=None):
        import types
        return types.MethodType(krea2_attn_forward_weight, obj)


class TextEncodeKrea2SysEditScaledAdvAttn(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="TextEncodeKrea2SysEditScaledAdvAttn",
            display_name="TextEncodeKrea2SysEditScaledAdvAttn",
            category="model/conditioning",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="Main text prompt. Supports weight syntax: (prompt:weight), e.g. (sunset:1.5).",
                ),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path).",
                ),
                io.Float.Input("multiplier", default=1.0, min=-1000.0, max=1000.0, step=0.1, tooltip="Overall multiplier applied to the final conditioning vector."),
                io.Float.Input("strength", default=1.0, min=0.0, max=4.0, step=0.05, tooltip="Global multiplier on the weighting effect. Effect compounds over all blocks."),
                io.Combo.Input(
                    "padding_method",
                    options=["zero-pad", "interpolate"],
                    default="zero-pad",
                    tooltip="Alignment method for images with different aspect ratios/resolutions.",
                ),
                io.String.Input(
                    "formula",
                    default="a",
                    multiline=False,
                    tooltip="Mathematical formula to blend conditioning outputs. Use variables a, b, c, d...",
                ),
                io.Autogrow.Input("image_inputs", template=autogrow_template),
            ],
            outputs=[
                io.Model.Output(),
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, clip, prompt, system_prompt, vlm_resolution, multiplier, strength, padding_method, formula, image_inputs: io.Autogrow.Type) -> io.NodeOutput:
        active_images = []
        if image_inputs is not None:
            for k, v in image_inputs.items():
                if v is not None:
                    active_images.append(v)

        # 1. Parse prompt weights using regex
        import re
        pattern = re.compile(r"\(([^():]+):(-?\d*\.?\d+)\)")
        terms = [(m.group(1).strip(), float(m.group(2))) for m in pattern.finditer(prompt)]
        clean_prompt = pattern.sub(lambda m: m.group(1), prompt)

        def process_vlm_image(image, res):
            if image is None:
                return None
            VLM_RESOLUTIONS = {
                "Fast (384)": 384,
                "Balanced (512)": 512,
                "Detailed (768)": 768,
                "Large (1024)": 1024,
                "X-Large (1280)": 1280,
                "XX-Large (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            if res == "Original":
                return image
            else:
                vlm_size = VLM_RESOLUTIONS[res]
                total_vlm = vlm_size * vlm_size
                scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                width_vlm = round(samples.shape[3] * scale_by_vlm)
                height_vlm = round(samples.shape[2] * scale_by_vlm)

                s_vlm = common_upscale(samples, width_vlm, height_vlm, "bicubic", "disabled")
                return s_vlm.movedim(1, -1)

        # 2. Get tokens mapping on clean prompt with representative (first) image or fallback
        if active_images:
            first_img = active_images[0]
            processed_first_img = process_vlm_image(first_img, vlm_resolution)

            modified_clean_prompt = clean_prompt
            if not any(tag in clean_prompt for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>", "image_input_"]):
                modified_clean_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + modified_clean_prompt

            if len(system_prompt) > 0:
                clean_full_prompt = (
                    "<|im_start|>user\n" + "<|im_end|>\n" +
                    "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                    "<|im_start|>user\n" + modified_clean_prompt + "<|im_end|>\n" +
                    "<|im_start|>assistant\n"
                )
            else:
                clean_full_prompt = (
                    "<|im_start|>user\n" + "<|im_end|>\n" +
                    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
                    "<|im_start|>user\n" + modified_clean_prompt + "<|im_end|>\n" +
                    "<|im_start|>assistant\n"
                )
            tok = clip.tokenize(clean_full_prompt, images=[processed_first_img], skip_template=True)
        else:
            if len(system_prompt) > 0:
                clean_full_prompt = (
                    "<|im_start|>user\n" + "<|im_end|>\n" +
                    "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                    "<|im_start|>user\n" + clean_prompt + "<|im_end|>\n" +
                    "<|im_start|>assistant\n"
                )
            else:
                clean_full_prompt = (
                    "<|im_start|>user\n" + "<|im_end|>\n" +
                    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
                    "<|im_start|>user\n" + clean_prompt + "<|im_end|>\n" +
                    "<|im_start|>assistant\n"
                )
            tok = clip.tokenize(clean_full_prompt, skip_template=True)

        key = next(iter(tok))
        token_list = tok[key][0]
        ids = []
        for t in token_list:
            if isinstance(t, tuple) and len(t) > 0:
                ids.append(t[0])
            elif isinstance(t, dict):
                ids.append(-1)
            else:
                ids.append(t)

        cond = clip.encode_from_tokens_scheduled(tok)
        cond_len = cond[0][0].shape[1]

        # Count text vs image tokens for mapping
        text_token_count = 0
        image_token_count = 0
        for t in token_list:
            if is_image_token(t):
                image_token_count += 1
            else:
                text_token_count += 1

        if image_token_count > 0:
            V = (cond_len - text_token_count) // image_token_count
        else:
            V = 1

        mapping = []
        current_idx = 0
        for t in token_list:
            is_img = is_image_token(t)
            size = V if is_img else 1
            start = current_idx
            end = current_idx + size
            mapping.append((start, end))
            current_idx = end

        start, end = _krea2_user_content_span(ids)
        if start is None:
            start, end = 0, len(ids)

        weight_pairs = []
        for phrase, w in terms:
            if w > 1.0:
                v_factor, k_bias = 1.0, strength * (w - 1.0) * 2.0
            else:
                v_factor, k_bias = 1.0 + strength * (w - 1.0), 0.0
            positions = []
            for variant in (" " + phrase, phrase):
                sub = _krea2_token_ids(clip, variant)
                ps, pe = _krea2_user_content_span(sub)
                if ps is not None:
                    sub = sub[ps:pe]
                matches = _find_subsequence(ids, sub, start, end)
                if matches:
                    for mi in matches:
                        for off in range(len(sub)):
                            t_idx = mi + off
                            if t_idx < len(mapping):
                                positions.append(mapping[t_idx][0])
                    break
            if not positions:
                import logging
                logging.warning(f"Krea2PromptWeight: phrase '{phrase}' not found in prompt; skipped.")
                continue
            for cp in positions:
                if 0 <= cp < cond_len:
                    weight_pairs.append((cp, v_factor, k_bias))

        # 3. Patch model
        model_clone = model.clone()
        if weight_pairs:
            import logging
            logging.info(f"Krea2PromptWeight (Attn): weighting {weight_pairs}")
            diffusion_model = model_clone.get_model_object("diffusion_model")
            transformer_options = model_clone.model_options.get("transformer_options", {}).copy()
            transformer_options["krea2_token_weights"] = weight_pairs
            model_clone.model_options["transformer_options"] = transformer_options

            for idx, block in enumerate(diffusion_model.blocks):
                if hasattr(block, "attn"):
                    patched_attn = Krea2WeightPatch().__get__(block.attn, block.attn.__class__)
                    model_clone.add_object_patch(f"diffusion_model.blocks.{idx}.attn.forward", patched_attn)

        # 4. Multpass encoding and blending
        if active_images:
            sequence_tensors = {}
            pooled_tensors = {}
            deepstack_dict = {}
            last_cond_dict = None

            for idx, img in enumerate(active_images):
                letter = chr(97 + idx)
                processed_img = process_vlm_image(img, vlm_resolution)

                modified_prompt = clean_prompt
                if not any(tag in clean_prompt for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>", "image_input_"]):
                    modified_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + modified_prompt

                if len(system_prompt) > 0:
                    full_prompt = (
                        "<|im_start|>user\n" + "<|im_end|>\n" +
                        "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                        "<|im_start|>user\n" + modified_prompt + "<|im_end|>\n" +
                        "<|im_start|>assistant\n"
                    )
                else:
                    full_prompt = (
                        "<|im_start|>user\n" + "<|im_end|>\n" +
                        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
                        "<|im_start|>user\n" + modified_prompt + "<|im_end|>\n" +
                        "<|im_start|>assistant\n"
                    )

                cond_X = encode_embedding_classical_scaled_bias(clip, full_prompt, images=[processed_img], skip_template=True)
                C_X = cond_X[0][0]
                P_X = cond_X[0][1].get("pooled_output", None)

                sequence_tensors[letter] = C_X
                if P_X is not None:
                    pooled_tensors[letter] = P_X

                if "embeds_info" in cond_X[0][1] and len(cond_X[0][1]["embeds_info"]) > 0:
                    extra = cond_X[0][1]["embeds_info"][0].get("extra", {})
                    if "deepstack" in extra:
                        deepstack_dict[letter] = extra["deepstack"]

                last_cond_dict = cond_X[0][1]

            C_blended, P_blended = evaluate_conditioning_formula(formula, sequence_tensors, pooled_tensors, padding_method=padding_method)

            deepstack_blended = None
            if deepstack_dict:
                first_key = next(iter(sequence_tensors.keys()))
                num_layers = len(deepstack_dict[first_key])
                max_vis_len = max(ds_list[0].shape[0] for ds_list in deepstack_dict.values())

                deepstack_blended = []
                import torch.nn.functional as F

                for l in range(num_layers):
                    layer_tensors = {let: ds_list[l] for let, ds_list in deepstack_dict.items()}

                    aligned_layer_tensors = {}
                    for name, tensor in layer_tensors.items():
                        if tensor.shape[0] < max_vis_len:
                            if padding_method == "interpolate":
                                tensor_perm = tensor.permute(1, 0).unsqueeze(0)
                                tensor_interp = F.interpolate(tensor_perm, size=max_vis_len, mode='linear', align_corners=False)
                                tensor = tensor_interp.squeeze(0).permute(1, 0)
                            else:
                                pad_size = max_vis_len - tensor.shape[0]
                                padding = torch.zeros((pad_size, tensor.shape[1]), device=tensor.device, dtype=tensor.dtype)
                                tensor = torch.cat([tensor, padding], dim=0)
                        aligned_layer_tensors[name] = tensor

                    safe_dict_layer = {
                        "__builtins__": {},
                        "clamp": torch.clamp,
                        "min": torch.minimum,
                        "max": torch.maximum,
                        "abs": torch.abs,
                    }
                    for name, t in aligned_layer_tensors.items():
                        safe_dict_layer[name] = t

                    layer_expression = re.sub(
                        r"\(\s*([a-zA-Z0-9_]+)\s*:\s*([0-9.-]+)\s*\)",
                        r"(\1 * \2)",
                        formula
                    )

                    try:
                        layer_blended = eval(layer_expression, safe_dict_layer, {})
                        deepstack_blended.append(layer_blended)
                    except Exception as e:
                        raise RuntimeError(f"Error evaluating DeepStack math expression at layer {l}: {e}")

            final_cond_dict = last_cond_dict.copy()
            if P_blended is not None:
                final_cond_dict["pooled_output"] = P_blended

            if deepstack_blended is not None and "embeds_info" in final_cond_dict and len(final_cond_dict["embeds_info"]) > 0:
                final_cond_dict["embeds_info"] = [final_cond_dict["embeds_info"][0].copy()]
                final_cond_dict["embeds_info"][0]["extra"] = final_cond_dict["embeds_info"][0]["extra"].copy()
                final_cond_dict["embeds_info"][0]["extra"]["deepstack"] = deepstack_blended

            if multiplier != 1.0:
                C_blended *= multiplier
                if "pooled_output" in final_cond_dict and final_cond_dict["pooled_output"] is not None:
                    final_cond_dict["pooled_output"] *= multiplier

            conditioning = [[C_blended, final_cond_dict]]
        else:
            conditioning = encode_embedding_classical_scaled_bias(clip, clean_full_prompt)
            if multiplier != 1.0:
                for i in range(len(conditioning)):
                    conditioning[i][0] *= multiplier
                    if "pooled_output" in conditioning[i][1] and conditioning[i][1]["pooled_output"] is not None:
                        conditioning[i][1]["pooled_output"] *= multiplier

        return io.NodeOutput(model_clone, conditioning)


