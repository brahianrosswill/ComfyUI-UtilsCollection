import re
import math
import torch
from enum import Enum

from comfy_api.latest import ComfyExtension, io
from comfy.utils import common_upscale

# -----------------------------------------------------------------------------
# 1. Helper classes, mappings and functions
# -----------------------------------------------------------------------------

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
        raise RuntimeError(f"Error evaluating textgen visual math expression '|{expression}|': {e}")

BlendConfig = io.Custom("BLEND_CONFIG")

def evaluate_image_consensus_blend(
    processed_images: dict,
    blend_config: dict = None,
    device: str = "cpu"
) -> torch.Tensor:
    if blend_config is None:
        blend_config = {"blend_preset": "off"}

    blend_preset = blend_config.get("blend_preset", "off")
    blend_method = blend_config.get("blend_method", "consensus")
    consensus_type = blend_config.get("consensus_type", "median")
    similarity_threshold = blend_config.get("similarity_threshold", 0.0)
    power_alpha = blend_config.get("power_alpha", 2.0)
    diversity_beta = blend_config.get("diversity_beta", 0.0)
    rescale_norm = blend_config.get("rescale_norm", False)
    global_scale = blend_config.get("global_scale", 1.0)

    presets = {
        "baseline": {"method": "consensus", "type": "median", "alpha": 2.0, "thresh": 0.0, "beta": 0.0, "scale": 1.0, "norm": False},
        "high_clarity": {"method": "consensus", "type": "median", "alpha": 3.0, "thresh": 0.3, "beta": 0.0, "scale": 1.0, "norm": False},
        "smooth": {"method": "consensus", "type": "mean", "alpha": 1.5, "thresh": 0.0, "beta": 0.0, "scale": 1.0, "norm": False},
        "varied_merge": {"method": "consensus", "type": "median", "alpha": 2.0, "thresh": 0.0, "beta": 0.0, "scale": 0.7, "norm": True},
        "diverse_concept": {"method": "consensus", "type": "median", "alpha": 2.0, "thresh": 0.0, "beta": 1.0, "scale": 0.7, "norm": True},
        "high_diversity_concept": {"method": "consensus", "type": "median", "alpha": 2.0, "thresh": 0.0, "beta": 2.0, "scale": 0.7, "norm": True}
    }

    if blend_preset != "off" and blend_preset in presets:
        p = presets[blend_preset]
        blend_method = p["method"]
        consensus_type = p["type"]
        power_alpha = p["alpha"]
        similarity_threshold = p["thresh"]
        diversity_beta = p["beta"]
        rescale_norm = p["norm"]
        user_global_scale = blend_config.get("global_scale", 1.0)
        if user_global_scale != 1.0:
            global_scale = user_global_scale
        else:
            global_scale = p["scale"]

    active_keys = sorted([k for k in processed_images.keys() if len(k) == 1 and k.islower()])
    tensors = [processed_images[k].to(device=device, dtype=torch.float32) for k in active_keys]

    if not tensors:
        return None

    stacked_images = torch.stack(tensors, dim=1) # [B, K, H, W, C]
    B, K, H, W, C = stacked_images.shape

    flattened = stacked_images.view(B, K, H*W, C)
    blended_flat_list = []

    for b in range(B):
        img_seq = flattened[b]

        if blend_method == "linear":
            merged = img_seq.mean(dim=0)
            if global_scale != 1.0:
                merged *= global_scale
            blended_flat_list.append(merged)
            continue

        if consensus_type == "median":
            consensus = torch.median(img_seq, dim=0).values
        else:
            consensus = img_seq.mean(dim=0)

        img_seq_norm = torch.nn.functional.normalize(img_seq, p=2, dim=2)
        consensus_norm = torch.nn.functional.normalize(consensus, p=2, dim=1)

        similarities = (img_seq_norm * consensus_norm.unsqueeze(0)).sum(dim=2) # [K, HW]

        weights = torch.zeros_like(similarities)
        mask = similarities >= similarity_threshold

        if diversity_beta > 0.0:
            weights_val = torch.pow(similarities, power_alpha) * torch.pow(1.001 - similarities, diversity_beta)
        else:
            weights_val = torch.pow(similarities, power_alpha)

        weights = torch.where(mask, weights_val, torch.zeros_like(weights_val))
        w_sum = weights.sum(dim=0, keepdim=True)
        weights = torch.where(w_sum > 0, weights / w_sum, torch.ones_like(weights) / K)

        merged_pixels = (img_seq * weights.unsqueeze(2)).sum(dim=0)

        if rescale_norm:
            avg_norm = torch.norm(img_seq, p=2, dim=2).mean(dim=0, keepdim=True).squeeze(0) # [HW]
            merged_norm = torch.norm(merged_pixels, p=2, dim=1) # [HW]
            scale_factor = torch.where(merged_norm > 0, avg_norm / merged_norm, torch.ones_like(merged_norm))
            merged_pixels *= scale_factor.unsqueeze(1)

        if global_scale != 1.0:
            merged_pixels *= global_scale

        blended_flat_list.append(merged_pixels)

    blended_flat = torch.stack(blended_flat_list, dim=0) # [B, HW, C]
    blended_image = blended_flat.view(B, H, W, C).to(dtype=tensors[0].dtype, device=tensors[0].device)

    return torch.clamp(blended_image, 0.0, 1.0)

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

# -----------------------------------------------------------------------------
# 2. Template Definitions for Unified Text Generation
# -----------------------------------------------------------------------------

MODEL_TEMPLATES = {
    "qwen35": {
        "system_prefix": "<|im_start|>system\n",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "assistant_prefix": "<|im_start|>assistant\n",
        "visual_token": "<|vision_start|><|image_pad|><|vision_end|>",
        "suppress_thinking": "<think>\n</think>\n"
    },
    "llama3": {
        "system_prefix": "<|start_header_id|>system<|end_header_id|>\n\n",
        "system_suffix": "<|eot_id|>",
        "user_prefix": "<|start_header_id|>user<|end_header_id|>\n\n",
        "user_suffix": "<|eot_id|>",
        "assistant_prefix": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "visual_token": "<image_soft_token>",
        "suppress_thinking": None
    },
    "gemma": {
        "system_prefix": "<start_of_turn>system\n",
        "system_suffix": "<end_of_turn>\n",
        "user_prefix": "<start_of_turn>user\n",
        "user_suffix": "<end_of_turn>\n",
        "assistant_prefix": "<start_of_turn>model\n",
        "visual_token": "<img><image_soft_token><end_of_image>",
        "suppress_thinking": "<think>\n\n</think>\n\n"
    },
    "custom": {
        "system_prefix": "",
        "system_suffix": "\n",
        "user_prefix": "\nUser: ",
        "user_suffix": "",
        "assistant_prefix": "\nAssistant: ",
        "visual_token": "<image>",
        "suppress_thinking": None
    }
}

# -----------------------------------------------------------------------------
# 3. Class Implementations
# -----------------------------------------------------------------------------

class UC_TextGenerate(io.ComfyNode):
    """
    Advanced text generation node that implements:
    - Multiple image handling via dynamic autogrow input.
    - Image rescaling limits to optimize performance and prevent out-of-memory.
    - Safe system prompt template selection for different standard formats.
    - Safe manual concatenation to prevent formatting errors with user characters.
    """
    @classmethod
    def define_schema(cls):
        # Sampling Mode Options
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

        # Dynamic Autogrow Setup for up to 16 images
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )

        return io.Schema(
            node_id="UC_TextGenerate",
            display_name="Advanced Text Generate (VLM & Multi-Image)",
            category="advanced/textgen",
            search_aliases=["LLM", "VLM", "textgen", "generate", "chat", "qwen", "gemma", "llama"],
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=False,
                    default="",
                    tooltip="Main query. Braces {} are fully safe. Supports visual blending formulas inside pipes, e.g. |(image_input_1 + image_input_2)/2|"
                ),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=False, default=""),
                io.Combo.Input(
                    "model_type",
                    options=list(MODEL_TEMPLATES.keys()),
                    default="qwen35",
                    tooltip="Configures the structure of delimiters, visual tokens, and thinking blocks."
                ),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Rescales connected image inputs to optimize performance and VRAM allocation."
                ),
                io.Int.Input("max_length", default=512, min=1, max=32768),
                io.DynamicCombo.Input("sampling_mode", options=sampling_options, display_name="Sampling Mode"),
                io.String.Input(
                    "formula",
                    default="",
                    multiline=False,
                    tooltip="Optional mathematical formula to blend image inputs at pixel-tensor level before encoding. Use variables a, b, c, d... to reference active connected image inputs. If empty, connected images are treated as separate inputs."
                ),
                io.Boolean.Input("thinking", optional=True, default=False, tooltip="Preserves chain-of-thought blocks if the model supports reasoning."),
                BlendConfig.Input("blend_config", optional=True, tooltip="Optional Consensus Blend Configuration input."),
                io.Autogrow.Input("image_inputs", template=autogrow_template),
            ],
            outputs=[
                io.String.Output(display_name="generated_text"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt, model_type, vlm_resolution, max_length, sampling_mode, formula="", thinking=False, blend_config: dict = None, image_inputs=None) -> io.NodeOutput:
        if clip is None:
            raise RuntimeError("ERROR: CLIP/TextEncoder input is invalid: None")

        # 1. Parse Autogrow Inputs
        raw_images = {}
        if image_inputs is not None:
            for k, v in image_inputs.items():
                if v is not None:
                    digits = re.findall(r'\d+', k)
                    idx = int(digits[0]) if digits else 1
                    raw_images[idx] = v

        is_zero_indexed = 0 in raw_images

        # 2. Rescale Images to Target Resolutions & Create Variable Mapping
        processed_images = {}
        for idx, (num, img) in enumerate(sorted(raw_images.items())):
            display_name = ImageInputMapping.get_display_name(num, is_zero_indexed)
            scaled_img = process_vlm_image(img, vlm_resolution)
            processed_images[display_name] = scaled_img

            # Map sequentially to letter variables a, b, c, d... (matching TextEncodeKrea2SystemEditScaledAdv)
            letter = chr(97 + idx) # 0 -> 'a', 1 -> 'b', 2 -> 'c'...
            processed_images[letter] = scaled_img

        # Ensure all image tensors are aligned to the same height and width if we have more than one image
        if len(raw_images) > 1:
            max_h = max(img.shape[1] for img in processed_images.values())
            max_w = max(img.shape[2] for img in processed_images.values())

            for key, img in list(processed_images.items()):
                if img.shape[1] != max_h or img.shape[2] != max_w:
                    # Move dimensions to [B, C, H, W] for interpolate
                    samples = img.movedim(-1, 1)
                    rescaled = torch.nn.functional.interpolate(
                        samples, size=(max_h, max_w), mode="bicubic", align_corners=False
                    )
                    # Move back to [B, H, W, C]
                    processed_images[key] = rescaled.movedim(1, -1)

        if blend_config is None:
            blend_config = {"blend_preset": "off"}
        blend_preset = blend_config.get("blend_preset", "off")

        # Determine if we should globally blend images
        if blend_preset != "off":
            try:
                import comfy
                device = comfy.model_management.get_torch_device()
                blended_image = evaluate_image_consensus_blend(
                    processed_images, blend_config=blend_config, device=device
                )
                # Override raw_images and processed_images to contain only this single blended image
                raw_images = {1: blended_image}
                processed_images = {
                    "image_input_1": blended_image,
                    "a": blended_image
                }
                is_zero_indexed = False
            except Exception as e:
                raise RuntimeError(f"Error evaluating image consensus blending: {e}")
        elif formula and formula.strip():
            try:
                blended_image = evaluate_formula(formula.strip(), processed_images)
                # Override raw_images and processed_images to contain only this single blended image
                raw_images = {1: blended_image}
                processed_images = {
                    "image_input_1": blended_image,
                    "a": blended_image
                }
                is_zero_indexed = False
            except Exception as e:
                raise RuntimeError(f"Error evaluating global textgen blending formula '{formula}': {e}")

        # 3. Parse and Evaluate Math Formulas inside |pipes|
        images_vl = []
        math_pattern = re.compile(r"\|([^|]+)\|")
        temp = MODEL_TEMPLATES[model_type]
        v_token = temp["visual_token"]

        def replace_formula(match):
            expression = match.group(1).strip()
            # Build list of valid active variables (e.g. "image_input_1", "a", "b", etc.)
            valid_vars = list(processed_images.keys())

            # Check if any valid variable is present as a word boundary inside the expression
            has_valid_var = any(re.search(r'\b' + re.escape(var) + r'\b', expression, re.IGNORECASE) for var in valid_vars)

            if not raw_images or not has_valid_var:
                return match.group(0)

            try:
                result_tensor = evaluate_formula(expression, processed_images)
                images_vl.append(result_tensor)
                return v_token
            except Exception:
                # Fallback to preserving the original text in case of evaluation error
                return match.group(0)

        modified_prompt = math_pattern.sub(replace_formula, prompt)

        # 4. Check for keyword references like 'image_input_1'
        pattern = re.compile(r'image_input_(\d+)', re.IGNORECASE)
        has_keywords = bool(pattern.search(modified_prompt)) or len(images_vl) > 0

        if has_keywords:
            def replace_keyword(match):
                num = int(match.group(1))
                dict_key = ImageInputMapping.get_dict_key(num, is_zero_indexed)
                if dict_key in raw_images:
                    img = raw_images[dict_key]
                    display_name = ImageInputMapping.get_display_name(dict_key, is_zero_indexed)
                    processed_img = processed_images.get(display_name, process_vlm_image(img, vlm_resolution))
                    images_vl.append(processed_img)
                    return v_token
                return ""
            modified_prompt = pattern.sub(replace_keyword, modified_prompt)
        else:
            # Fallback behavior: prepend all connected images in index-sorted order
            image_prompt_prefix = ""
            for num in sorted(raw_images.keys()):
                display_name = ImageInputMapping.get_display_name(num, is_zero_indexed)
                processed_img = processed_images[display_name]
                images_vl.append(processed_img)
                image_prompt_prefix += v_token
            modified_prompt = image_prompt_prefix + modified_prompt

        # 5. Build Safe Non-Formatting Chat Template
        # We completely avoid standard formatting or f-strings which fail if prompts contain { } braces.
        full_prompt = ""

        # System prompt segment
        if system_prompt and system_prompt.strip():
            full_prompt += temp["system_prefix"] + system_prompt + temp["system_suffix"]

        # User query segment
        full_prompt += temp["user_prefix"] + modified_prompt + temp["user_suffix"]

        # Assistant block entry
        full_prompt += temp["assistant_prefix"]

        # If thinking is disabled, explicitly append reasoning suppression sequences if the model template has them
        if not thinking and temp["suppress_thinking"]:
            full_prompt += temp["suppress_thinking"]

        # 6. Setup Tokenizer Dispatch Arguments
        kwargs = {}
        if len(images_vl) > 0:
            kwargs["images"] = images_vl
            if len(images_vl) == 1:
                kwargs["image"] = images_vl[0]

        # Use skip_template=True because we have assembled perfect, model-specific delimiters ourselves.
        tokens = clip.tokenize(
            full_prompt,
            skip_template=True,
            min_length=1,
            thinking=thinking,
            **kwargs
        )

        # 7. Extract Parameters from sampling_mode dynamic combo
        do_sample = sampling_mode.get("sampling_mode") == "on"
        temperature = sampling_mode.get("temperature", 1.0)
        top_k = sampling_mode.get("top_k", 50)
        top_p = sampling_mode.get("top_p", 1.0)
        min_p = sampling_mode.get("min_p", 0.0)
        seed = sampling_mode.get("seed", None)
        repetition_penalty = sampling_mode.get("repetition_penalty", 1.0)
        presence_penalty = sampling_mode.get("presence_penalty", 0.0)

        # 8. Generation & Decoding
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
            seed=seed
        )

        generated_text = clip.decode(generated_ids, skip_special_tokens=True)
        return io.NodeOutput(generated_text)


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
            [<|im_start|>system\\n{system_message}<|im_end|>\\n]   ... optional
            <|im_start|>user\\n
            [<|vision_start|><|image_pad|><|vision_end|>]          ... if image
            {prompt}<|im_end|>\\n
            <|im_start|>assistant\\n
            [<think>\\n</think>\\n]                                  ... if not thinking
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
