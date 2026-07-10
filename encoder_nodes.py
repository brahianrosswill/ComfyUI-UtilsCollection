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

BlendConfig = io.Custom("BLEND_CONFIG")

class UC_ConsensusBlendConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_ConsensusBlendConfig",
            display_name="Consensus Blend Configurator",
            category="advanced/conditioning",
            inputs=[
                io.Combo.Input(
                    "blend_preset",
                    options=[
                        "off", "custom", "baseline", "high_clarity", "smooth", "varied_merge", "diverse_concept", "high_diversity_concept",
                        "dsc_baseline", "dsc_high_clarity", "dsc_smooth", "dsc_varied_merge", "dsc_diverse_concept", "dsc_high_diversity_concept"
                    ],
                    default="baseline",
                    tooltip="Preset configuration for Consensus-Weighted Blending. Set to 'off' to use normal formula blending, or 'custom' to use the manual parameter overrides below. Other presets WILL ignore and override the manual sliders below."
                ),
                io.Combo.Input(
                    "blend_method",
                    options=["linear", "consensus"],
                    default="consensus",
                    tooltip="Only active when blend_preset is 'custom'. 'consensus' aligns inputs dynamically and filters noise; 'linear' performs simple averaging."
                ),
                io.Combo.Input(
                    "consensus_type",
                    options=["mean", "median"],
                    default="median",
                    tooltip="Only active when blend_preset is 'custom'. 'median' completely rejects up to 50% outlying features; 'mean' provides smooth averaging."
                ),
                io.Combo.Input(
                    "alignment_method",
                    options=["index", "similarity"],
                    default="similarity",
                    tooltip="Only active when blend_preset is 'custom'. 'similarity' matches tokens in the embedding space via cosine similarity; 'index' aligns dimensions sequentially."
                ),
                io.Float.Input("alignment_threshold", default=0.4, min=0.0, max=1.0, step=0.01, tooltip="Only active when blend_preset is 'custom' and alignment_method is 'similarity'. Minimum similarity required to match two tokens."),
                io.Float.Input("similarity_threshold", default=0.0, min=-1.0, max=1.0, step=0.01, tooltip="Only active when blend_preset is 'custom'. Prunes tokens from individual passes if similarity to the consensus falls below this limit."),
                io.Float.Input("power_alpha", default=2.0, min=0.0, max=10.0, step=0.1, tooltip="Only active when blend_preset is 'custom'. Soft-masking exponent. Higher values penalize outlying elements heavily (e.g., 2.0)."),
                io.Float.Input("diversity_beta", default=0.0, min=0.0, max=10.0, step=0.1, tooltip="Only active when blend_preset is 'custom'. Diversity exponent. Values > 0.0 damp overfitted features and boost unique details (e.g., 1.5)."),
                io.Boolean.Input("rescale_norm", default=True, tooltip="Only active when blend_preset is 'custom'. Rescales vector magnitudes to maintain prompt activation energy and prevent desaturation collapse."),
                io.Float.Input("global_scale", default=1.0, min=0.0, max=10.0, step=0.01, tooltip="Always active (works for custom and presets). Scaling factor applied to the final merged outputs. Set to 1.25+ to force single-subject convergence."),
                io.Boolean.Input("dynamic_similarity_contrast", default=False, tooltip="Only active when blend_preset is 'custom'. Maps similarities to a soft [0.7, 1.0] band to prevent WTA collapse while boosting blending contrast."),
                io.Boolean.Input("soft_comfort_bandpass", default=False, tooltip="Only active when blend_preset is 'custom'. Relocates the bandpass penalty ceiling to a soft 1.5 coordinate to prevent outlier overshoots."),
                io.Boolean.Input("isolate_visual_tokens", default=True, tooltip="Always active (works for custom and presets). Isolates visual embeddings and blends them spatially to prevent the Zero-Blending Gap on disparate concepts."),
                io.Combo.Input(
                    "visual_blend_method",
                    options=["index-consensus", "similarity-consensus", "linear", "off"],
                    default="index-consensus",
                    tooltip="Always active (works for custom and presets). Method used to blend isolated visual tokens. 'index-consensus' is highly recommended as it blends them spatially."
                ),
                io.Combo.Input(
                    "preserve_text_pass",
                    options=["reference", "blend"],
                    default="reference",
                    tooltip="Always active (works for custom and presets). 'reference' keeps the first pass prompt 100% sharp to avoid text dilution; 'blend' aligns and averages them."
                )
            ],
            outputs=[
                BlendConfig.Output("config")
            ]
        )

    @classmethod
    def execute(
        cls,
        blend_preset: str,
        blend_method: str,
        consensus_type: str,
        alignment_method: str,
        alignment_threshold: float,
        similarity_threshold: float,
        power_alpha: float,
        diversity_beta: float,
        rescale_norm: bool,
        global_scale: float,
        dynamic_similarity_contrast: bool = False,
        soft_comfort_bandpass: bool = False,
        isolate_visual_tokens: bool = True,
        visual_blend_method: str = "index-consensus",
        preserve_text_pass: str = "reference"
    ) -> io.NodeOutput:
        config = {
            "blend_preset": blend_preset,
            "blend_method": blend_method,
            "consensus_type": consensus_type,
            "alignment_method": alignment_method,
            "alignment_threshold": alignment_threshold,
            "similarity_threshold": similarity_threshold,
            "power_alpha": power_alpha,
            "diversity_beta": diversity_beta,
            "rescale_norm": rescale_norm,
            "global_scale": global_scale,
            "dynamic_similarity_contrast": dynamic_similarity_contrast,
            "soft_comfort_bandpass": soft_comfort_bandpass,
            "isolate_visual_tokens": isolate_visual_tokens,
            "visual_blend_method": visual_blend_method,
            "preserve_text_pass": preserve_text_pass
        }
        return io.NodeOutput(config)

def evaluate_conditioning_consensus_blend(
    sequence_tensors: dict,
    pooled_tensors: dict,
    blend_config: dict = None,
    device: str = "cpu",
    visual_ranges: dict = None
) -> tuple:
    if blend_config is None:
        blend_config = {"blend_preset": "off"}

    blend_preset = blend_config.get("blend_preset", "off")
    blend_method = blend_config.get("blend_method", "consensus")
    consensus_type = blend_config.get("consensus_type", "median")
    alignment_method = blend_config.get("alignment_method", "similarity")
    alignment_threshold = blend_config.get("alignment_threshold", 0.4)
    similarity_threshold = blend_config.get("similarity_threshold", 0.0)
    power_alpha = blend_config.get("power_alpha", 2.0)
    diversity_beta = blend_config.get("diversity_beta", 0.0)
    rescale_norm = blend_config.get("rescale_norm", True)
    global_scale = blend_config.get("global_scale", 1.0)

    presets = {
        # Absolute Spec Presets (Original)
        "baseline": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.0, "thresh": 0.0, "beta": 0.0, "scale": 1.0, "norm": False, "dsc": False, "soft_comfort": False},
        "high_clarity": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 3.0, "thresh": 0.3, "beta": 0.0, "scale": 1.0, "norm": False, "dsc": False, "soft_comfort": False},
        "smooth": {"method": "consensus", "type": "mean", "align": "similarity", "alpha": 1.5, "thresh": 0.0, "beta": 0.0, "scale": 1.0, "norm": False, "dsc": False, "soft_comfort": False},
        "varied_merge": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.0, "thresh": 0.0, "beta": 0.0, "scale": 0.7, "norm": True, "dsc": False, "soft_comfort": False},
        "diverse_concept": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.0, "thresh": 0.0, "beta": 1.0, "scale": 0.7, "norm": True, "dsc": False, "soft_comfort": False},
        "high_diversity_concept": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.0, "thresh": 0.0, "beta": 2.0, "scale": 0.7, "norm": True, "dsc": False, "soft_comfort": False},

        # Dynamic Contrast Presets (New)
        "dsc_baseline": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.0, "thresh": 0.0, "beta": 0.0, "scale": 1.0, "norm": False, "dsc": True, "soft_comfort": True},
        "dsc_high_clarity": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 4.0, "thresh": 0.3, "beta": 0.0, "scale": 1.0, "norm": False, "dsc": True, "soft_comfort": True},
        "dsc_smooth": {"method": "consensus", "type": "mean", "align": "similarity", "alpha": 1.0, "thresh": 0.0, "beta": 0.0, "scale": 1.0, "norm": False, "dsc": True, "soft_comfort": True},
        "dsc_varied_merge": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.5, "thresh": 0.0, "beta": 0.0, "scale": 0.7, "norm": True, "dsc": True, "soft_comfort": True},
        "dsc_diverse_concept": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.0, "thresh": 0.0, "beta": 1.5, "scale": 0.7, "norm": True, "dsc": True, "soft_comfort": True},
        "dsc_high_diversity_concept": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.0, "thresh": 0.0, "beta": 3.0, "scale": 0.7, "norm": True, "dsc": True, "soft_comfort": True}
    }

    dsc_enabled = False
    soft_comfort_enabled = False

    if blend_preset != "off" and blend_preset in presets:
        p = presets[blend_preset]
        blend_method = p["method"]
        consensus_type = p["type"]
        alignment_method = p["align"]
        power_alpha = p["alpha"]
        similarity_threshold = p["thresh"]
        diversity_beta = p["beta"]
        rescale_norm = p["norm"]
        dsc_enabled = p.get("dsc", False)
        soft_comfort_enabled = p.get("soft_comfort", False)
        user_global_scale = blend_config.get("global_scale", 1.0)
        if user_global_scale != 1.0:
            global_scale = user_global_scale
        else:
            global_scale = p["scale"]
    elif blend_preset == "custom":
        dsc_enabled = blend_config.get("dynamic_similarity_contrast", False)
        soft_comfort_enabled = blend_config.get("soft_comfort_bandpass", False)

    active_keys = sorted(list(sequence_tensors.keys()))
    tensors_list = [sequence_tensors[k] for k in active_keys]

    if not tensors_list:
        return None, None

    B = tensors_list[0].shape[0]
    D = tensors_list[0].shape[2]

    import logging
    logging.info(f"--- CWB START: keys={active_keys}, B={B}, D={D} ---")
    for k in active_keys:
        logging.info(f"  Input '{k}' shape={sequence_tensors[k].shape}")

    isolate_visual_tokens = blend_config.get("isolate_visual_tokens", True)
    visual_blend_method = blend_config.get("visual_blend_method", "index-consensus")
    preserve_text_pass = blend_config.get("preserve_text_pass", "reference")

    if isolate_visual_tokens and visual_ranges and any(r is not None and r != (0, 0) for r in visual_ranges.values()):
        C_blended_list = []
        for b in range(B):
            batch_tensors_dict = {k: sequence_tensors[k][b].to(device=device, dtype=torch.float32) for k in active_keys}

            ref_key = active_keys[0]
            ref_range = visual_ranges.get(ref_key, (0, 0))
            if ref_range is None:
                ref_range = (0, 0)

            prefixes = {}
            visuals = {}
            suffixes = {}

            for k in active_keys:
                t = batch_tensors_dict[k]
                vr = visual_ranges.get(k, (0, 0))
                if vr is None or vr == (0, 0):
                    prefixes[k] = t
                    visuals[k] = torch.zeros((0, D), device=device, dtype=t.dtype)
                    suffixes[k] = torch.zeros((0, D), device=device, dtype=t.dtype)
                else:
                    v_start, v_end = vr
                    prefixes[k] = t[:v_start, :]
                    visuals[k] = t[v_start:v_end, :]
                    suffixes[k] = t[v_end:, :]

            # --- 1. Blend Visuals ---
            active_visuals = {k: v for k, v in visuals.items() if v.shape[0] > 0}
            if active_visuals:
                max_vis_len = max(v.shape[0] for v in active_visuals.values())
                aligned_visuals = {}
                for k, v in active_visuals.items():
                    if v.shape[0] != max_vis_len:
                        v_perm = v.permute(1, 0).unsqueeze(0) # [1, D, N]
                        v_interp = torch.nn.functional.interpolate(v_perm, size=max_vis_len, mode='linear', align_corners=False)
                        v = v_interp.squeeze(0).permute(1, 0) # [max_vis_len, D]
                    aligned_visuals[k] = v

                wrapped_visuals = {k: v.unsqueeze(0) for k, v in aligned_visuals.items()}

                vis_blend_config = blend_config.copy()
                if visual_blend_method == "index-consensus":
                    vis_blend_config["blend_method"] = "consensus"
                    vis_blend_config["alignment_method"] = "index"
                elif visual_blend_method == "similarity-consensus":
                    vis_blend_config["blend_method"] = "consensus"
                    vis_blend_config["alignment_method"] = "similarity"
                elif visual_blend_method == "linear":
                    vis_blend_config["blend_method"] = "linear"
                elif visual_blend_method == "off":
                    vis_blend_config = None

                if vis_blend_config is not None:
                    vis_blend_config["isolate_visual_tokens"] = False
                    blended_vis_seq, _ = evaluate_conditioning_consensus_blend(
                        wrapped_visuals, {}, blend_config=vis_blend_config, device=device
                    )
                    blended_vis_2d = blended_vis_seq.squeeze(0)
                else:
                    blended_vis_2d = aligned_visuals[ref_key]
            else:
                blended_vis_2d = torch.zeros((0, D), device=device)

            # --- 2. Blend Prefixes and Suffixes ---
            if preserve_text_pass == "reference":
                blended_prefix = prefixes[ref_key]
                blended_suffix = suffixes[ref_key]
            else:
                wrapped_prefixes = {k: v.unsqueeze(0) for k, v in prefixes.items() if v.shape[0] > 0}
                if wrapped_prefixes:
                    prefix_blend_config = blend_config.copy()
                    prefix_blend_config["alignment_method"] = "similarity"
                    prefix_blend_config["isolate_visual_tokens"] = False
                    blended_pref_seq, _ = evaluate_conditioning_consensus_blend(
                        wrapped_prefixes, {}, blend_config=prefix_blend_config, device=device
                    )
                    blended_prefix = blended_pref_seq.squeeze(0)
                else:
                    blended_prefix = torch.zeros((0, D), device=device)

                wrapped_suffixes = {k: v.unsqueeze(0) for k, v in suffixes.items() if v.shape[0] > 0}
                if wrapped_suffixes:
                    suffix_blend_config = blend_config.copy()
                    suffix_blend_config["alignment_method"] = "similarity"
                    suffix_blend_config["isolate_visual_tokens"] = False
                    blended_suff_seq, _ = evaluate_conditioning_consensus_blend(
                        wrapped_suffixes, {}, blend_config=suffix_blend_config, device=device
                    )
                    blended_suffix = blended_suff_seq.squeeze(0)
                else:
                    blended_suffix = torch.zeros((0, D), device=device)

            # --- 3. Stitch them back together ---
            final_seq = torch.cat([blended_prefix, blended_vis_2d, blended_suffix], dim=0)
            C_blended_list.append(final_seq)

        C_blended = torch.stack(C_blended_list, dim=0).to(dtype=tensors_list[0].dtype, device=tensors_list[0].device)
    else:
        C_blended_list = []

        for b in range(B):
            batch_tensors = [t[b].to(device=device, dtype=torch.float32) for t in tensors_list]

            if blend_method == "linear":
                max_len = max(t.shape[0] for t in batch_tensors)
                padded = []
                for t in batch_tensors:
                    if t.shape[0] < max_len:
                        padding = torch.zeros((max_len - t.shape[0], D), device=device, dtype=t.dtype)
                        t = torch.cat([t, padding], dim=0)
                    padded.append(t)
                stacked = torch.stack(padded, dim=0)
                merged_seq = torch.mean(stacked, dim=0)
                if global_scale != 1.0:
                    merged_seq *= global_scale
                C_blended_list.append(merged_seq)
                continue

            if alignment_method == "similarity":
                ref_idx = max(range(len(batch_tensors)), key=lambda idx: batch_tensors[idx].shape[0])
                ref_tensor = batch_tensors[ref_idx]
                N_ref = ref_tensor.shape[0]

                ref_norm = torch.nn.functional.normalize(ref_tensor, p=2, dim=1)
                aligned_groups = [[] for _ in range(N_ref)]

                for idx, t in enumerate(batch_tensors):
                    N_k = t.shape[0]
                    if idx == ref_idx:
                        for i in range(N_ref):
                            aligned_groups[i].append(t[i])
                        continue

                    t_norm = torch.nn.functional.normalize(t, p=2, dim=1)
                    sim_matrix = torch.mm(ref_norm, t_norm.t())

                    matched_tk_idx = [-1] * N_ref
                    sim_matrix_tmp = sim_matrix.clone()

                    for _ in range(min(N_ref, N_k)):
                        flat_idx = torch.argmax(sim_matrix_tmp)
                        max_val = sim_matrix_tmp.flatten()[flat_idx].item()

                        if max_val < alignment_threshold:
                            break

                        r_idx = flat_idx // N_k
                        c_idx = flat_idx % N_k

                        matched_tk_idx[r_idx.item()] = c_idx.item()

                        sim_matrix_tmp[r_idx, :] = -100.0
                        sim_matrix_tmp[:, c_idx] = -100.0

                    for r_idx in range(N_ref):
                        matched_c = matched_tk_idx[r_idx]
                        if matched_c != -1:
                            aligned_groups[r_idx].append(t[matched_c])

                merged_seq = torch.zeros((N_ref, D), dtype=torch.float32, device=device)
                for r_idx in range(N_ref):
                    row_tensors = aligned_groups[r_idx]
                    if not row_tensors:
                        continue
                    stacked = torch.stack(row_tensors, dim=0)

                    if consensus_type == "median":
                        consensus = torch.median(stacked, dim=0).values
                    else:
                        consensus = torch.mean(stacked, dim=0)

                    stacked_norm = torch.nn.functional.normalize(stacked, p=2, dim=1)
                    consensus_norm = torch.nn.functional.normalize(consensus, p=2, dim=0)
                    similarities = torch.mv(stacked_norm, consensus_norm)

                    # Dynamic similarity contrast stretching (boosts distinction among high-similarity tokens)
                    if dsc_enabled:
                        min_sim = similarities.min()
                        max_sim = similarities.max()
                        if max_sim > min_sim:
                            # Map range [min_sim, max_sim] to a soft, healthy [0.7, 1.0] band
                            # This prevents "Winner-Takes-All" collapse while still providing pronounced, beautiful blending.
                            stretched_sims = 0.7 + 0.3 * (similarities - min_sim) / (max_sim - min_sim + 1e-8)
                        else:
                            stretched_sims = similarities
                    else:
                        stretched_sims = similarities

                    row_weights = torch.zeros_like(similarities)
                    mask = similarities >= similarity_threshold

                    if mask.any():
                        if diversity_beta > 0.0:
                            distance_base = 1.5 if soft_comfort_enabled else 1.001
                            row_weights[mask] = torch.pow(stretched_sims[mask], power_alpha) * torch.pow(distance_base - stretched_sims[mask], diversity_beta)
                        else:
                            row_weights[mask] = torch.pow(stretched_sims[mask], power_alpha)
                        w_sum = row_weights.sum()
                        if w_sum > 0:
                            row_weights /= w_sum
                        else:
                            row_weights = torch.ones_like(similarities) / len(similarities)
                    else:
                        row_weights = torch.ones_like(similarities) / len(similarities)

                    merged_vec = (stacked * row_weights.unsqueeze(1)).sum(dim=0)

                    if rescale_norm:
                        avg_norm = torch.norm(stacked, p=2, dim=1).mean()
                        merged_norm = torch.norm(merged_vec, p=2)
                        if merged_norm > 0:
                            merged_vec = (merged_vec / merged_norm) * avg_norm

                    if global_scale != 1.0:
                        merged_vec *= global_scale
                    merged_seq[r_idx] = merged_vec
                C_blended_list.append(merged_seq)
            else:
                # Index-based matching
                max_len = max(t.shape[0] for t in batch_tensors)
                merged_seq = torch.zeros((max_len, D), dtype=torch.float32, device=device)
                for i in range(max_len):
                    row_tensors = []
                    for t in batch_tensors:
                        if t.shape[0] > i:
                            row_tensors.append(t[i])
                    if not row_tensors:
                        continue
                    stacked = torch.stack(row_tensors, dim=0)

                    if consensus_type == "median":
                        consensus = torch.median(stacked, dim=0).values
                    else:
                        consensus = torch.mean(stacked, dim=0)

                    stacked_norm = torch.nn.functional.normalize(stacked, p=2, dim=1)
                    consensus_norm = torch.nn.functional.normalize(consensus, p=2, dim=0)
                    similarities = torch.mv(stacked_norm, consensus_norm)

                    # Dynamic similarity contrast stretching (boosts distinction among high-similarity tokens)
                    if dsc_enabled:
                        min_sim = similarities.min()
                        max_sim = similarities.max()
                        if max_sim > min_sim:
                            # Map range [min_sim, max_sim] to a soft, healthy [0.7, 1.0] band
                            # This prevents "Winner-Takes-All" collapse while still providing pronounced, beautiful blending.
                            stretched_sims = 0.7 + 0.3 * (similarities - min_sim) / (max_sim - min_sim + 1e-8)
                        else:
                            stretched_sims = similarities
                    else:
                        stretched_sims = similarities

                    row_weights = torch.zeros_like(similarities)
                    mask = similarities >= similarity_threshold

                    if mask.any():
                        if diversity_beta > 0.0:
                            distance_base = 1.5 if soft_comfort_enabled else 1.001
                            row_weights[mask] = torch.pow(stretched_sims[mask], power_alpha) * torch.pow(distance_base - stretched_sims[mask], diversity_beta)
                        else:
                            row_weights[mask] = torch.pow(stretched_sims[mask], power_alpha)
                        w_sum = row_weights.sum()
                        if w_sum > 0:
                            row_weights /= w_sum
                        else:
                            row_weights = torch.ones_like(similarities) / len(similarities)
                    else:
                        row_weights = torch.ones_like(similarities) / len(similarities)

                    merged_vec = (stacked * row_weights.unsqueeze(1)).sum(dim=0)

                    if rescale_norm:
                        avg_norm = torch.norm(stacked, p=2, dim=1).mean()
                        merged_norm = torch.norm(merged_vec, p=2)
                        if merged_norm > 0:
                            merged_vec = (merged_vec / merged_norm) * avg_norm

                    if global_scale != 1.0:
                        merged_vec *= global_scale
                    merged_seq[i] = merged_vec
                C_blended_list.append(merged_seq)

        C_blended = torch.stack(C_blended_list, dim=0).to(dtype=tensors_list[0].dtype, device=tensors_list[0].device)

    # 3. Blend pooled tensors (alignment is index-based as length is always 1)
    P_blended = None
    if any(p is not None for p in pooled_tensors.values()):
        pooled_list_active = [pooled_tensors[k] for k in active_keys if pooled_tensors.get(k) is not None]
        stacked_pooled = torch.stack(pooled_list_active, dim=1) # [B, K, D]
        P_blended_batches = []
        for b in range(B):
            stacked_p = stacked_pooled[b] # [K, D]
            if consensus_type == "median":
                consensus_p = torch.median(stacked_p, dim=0).values
            else:
                consensus_p = torch.mean(stacked_p, dim=0)

            stacked_p_norm = torch.nn.functional.normalize(stacked_p, p=2, dim=1)
            consensus_p_norm = torch.nn.functional.normalize(consensus_p, p=2, dim=0)
            similarities_p = torch.mv(stacked_p_norm, consensus_p_norm)

            # Dynamic similarity contrast stretching (boosts distinction among high-similarity tokens)
            if dsc_enabled:
                min_sim_p = similarities_p.min()
                max_sim_p = similarities_p.max()
                if max_sim_p > min_sim_p:
                    # Map range [min_sim_p, max_sim_p] to a soft, healthy [0.7, 1.0] band
                    # This prevents "Winner-Takes-All" collapse while still providing pronounced, beautiful blending.
                    stretched_sims_p = 0.7 + 0.3 * (similarities_p - min_sim_p) / (max_sim_p - min_sim_p + 1e-8)
                else:
                    stretched_sims_p = similarities_p
            else:
                stretched_sims_p = similarities_p

            weights_p = torch.zeros_like(similarities_p)
            mask_p = similarities_p >= similarity_threshold

            if mask_p.any():
                if diversity_beta > 0.0:
                    distance_base = 1.5 if soft_comfort_enabled else 1.001
                    weights_p[mask_p] = torch.pow(stretched_sims_p[mask_p], power_alpha) * torch.pow(distance_base - stretched_sims_p[mask_p], diversity_beta)
                else:
                    weights_p[mask_p] = torch.pow(stretched_sims_p[mask_p], power_alpha)
                w_sum_p = weights_p.sum()
                if w_sum_p > 0:
                    weights_p /= w_sum_p
                else:
                    weights_p = torch.ones_like(similarities_p) / len(similarities_p)
            else:
                weights_p = torch.ones_like(similarities_p) / len(similarities_p)

            merged_p = (stacked_p * weights_p.unsqueeze(1)).sum(dim=0)
            if rescale_norm:
                avg_norm_p = torch.norm(stacked_p, p=2, dim=1).mean()
                merged_p_norm = torch.norm(merged_p, p=2)
                if merged_p_norm > 0:
                    merged_p = (merged_p / merged_p_norm) * avg_norm_p
            if global_scale != 1.0:
                merged_p *= global_scale
            P_blended_batches.append(merged_p)
        P_blended = torch.stack(P_blended_batches, dim=0).to(dtype=tensors_list[0].dtype, device=tensors_list[0].device)

    logging.info(f"--- CWB END: C_blended shape={C_blended.shape}, P_blended shape={P_blended.shape if P_blended is not None else None} ---")
    return C_blended, P_blended

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

        conditioning = encode_embedding_scaled_bias(clip, prompt, llama_template=llama_template, skip_template=True)
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
        skip_template = False
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
            skip_template = True
        elif len(system_prompt) > 0:
            template = SYSTEM_PROMPT_TEMPLATES.get(model_type, SYSTEM_PROMPT_TEMPLATES["flux2dev"])
            llama_template = f"{template['prefix']}{system_prompt}{template['suffix']}"
            skip_template = (model_type == "klein")
        else:
            llama_template = None

        conditioning = encode_embedding_scaled_bias(clip, prompt, llama_template=llama_template, skip_template=skip_template)
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

        tokens = clip.tokenize(prompt, llama_template=llama_template, skip_template=True)
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
            tokens = clip.tokenize(prompt, llama_template=llama_template, skip_template=True)
        elif len(system_prompt) > 0:
            template = SYSTEM_PROMPT_TEMPLATES.get(model_type, SYSTEM_PROMPT_TEMPLATES["flux2dev"])
            llama_template = f"{template['prefix']}{system_prompt}{template['suffix']}"
            # If klein was chosen but without custom thinking_content, it uses the SYSTEM_PROMPT_TEMPLATES definition
            # which has an empty thinking block pre-defined inside suffix: "<think>\n\n</think>\n\n".
            # We must skip template to prevent the core from appending another redundant think block!
            skip_template = (model_type == "klein")
            tokens = clip.tokenize(prompt, llama_template=llama_template, skip_template=skip_template)
        else:
            tokens = clip.tokenize(prompt)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)




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
                BlendConfig.Input("blend_config", optional=True, tooltip="Optional Consensus Blend Configuration input."),
                io.Autogrow.Input("image_inputs", template=autogrow_template),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt, vlm_resolution, multiplier, padding_method, formula, image_inputs: io.Autogrow.Type, blend_config: dict = None) -> io.NodeOutput:
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
        visual_ranges = {}
        ref_cond_dict = None
        last_cond_dict = None

        if blend_config is None:
            blend_config = {"blend_preset": "off"}
        blend_preset = blend_config.get("blend_preset", "off")

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

            # Tokenize and find visual token range for isolation blending
            try:
                tokens = clip.tokenize(full_prompt, images=[processed_img], skip_template=True)
                vis_start, vis_end = find_visual_token_range(tokens, C_X)
                visual_ranges[letter] = (vis_start, vis_end)
            except Exception:
                visual_ranges[letter] = (0, 0)

            # Extract DeepStack per-layer tensors if present
            if "embeds_info" in cond_X[0][1] and len(cond_X[0][1]["embeds_info"]) > 0:
                extra = cond_X[0][1]["embeds_info"][0].get("extra", {})
                if "deepstack" in extra:
                    deepstack_dict[letter] = extra["deepstack"]

            if idx == 0:
                ref_cond_dict = cond_X[0][1]
            last_cond_dict = cond_X[0][1]

        # Evaluate mathematical formula or consensus on sequence and pooled tensors
        if blend_preset != "off":
            import comfy
            device = comfy.model_management.get_torch_device()
            C_blended, P_blended = evaluate_conditioning_consensus_blend(
                sequence_tensors, pooled_tensors, blend_config=blend_config, device=device, visual_ranges=visual_ranges
            )
        else:
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

                if blend_preset != "off":
                    wrapped_layer_tensors = {let: t.unsqueeze(0) for let, t in layer_tensors.items()}
                    import comfy
                    device = comfy.model_management.get_torch_device()
                    # DeepStack intermediate layers must be blended purely using similarity weights
                    # and must NEVER carry global scales or vector magnitude norm rescalings which distort inner residuals.
                    ds_blend_config = blend_config.copy()
                    ds_blend_config["global_scale"] = 1.0
                    ds_blend_config["rescale_norm"] = False
                    ds_blend_config["alignment_method"] = "index"
                    C_l_blended, _ = evaluate_conditioning_consensus_blend(
                        wrapped_layer_tensors, {}, blend_config=ds_blend_config, device=device
                    )
                    deepstack_blended.append(C_l_blended.squeeze(0))
                else:
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
                BlendConfig.Input("blend_config", optional=True, tooltip="Optional Consensus Blend Configuration input."),
                io.Autogrow.Input("image_inputs", template=autogrow_template),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vlm_resolution, multiplier, padding_method, formula, image_inputs: io.Autogrow.Type, blend_config: dict = None) -> io.NodeOutput:
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
        visual_ranges = {}
        ref_cond_dict = None
        last_cond_dict = None

        if blend_config is None:
            blend_config = {"blend_preset": "off"}
        blend_preset = blend_config.get("blend_preset", "off")

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

            # Tokenize and find visual token range for isolation blending
            try:
                tokens = clip.tokenize(modified_prompt, images=[processed_img], skip_template=True)
                vis_start, vis_end = find_visual_token_range(tokens, C_X)
                visual_ranges[letter] = (vis_start, vis_end)
            except Exception:
                visual_ranges[letter] = (0, 0)

            if idx == 0:
                ref_cond_dict = cond_X[0][1]
            last_cond_dict = cond_X[0][1]

        # Evaluate mathematical formula or consensus on sequence and pooled tensors
        if blend_preset != "off":
            import comfy
            device = comfy.model_management.get_torch_device()
            C_blended, P_blended = evaluate_conditioning_consensus_blend(
                sequence_tensors, pooled_tensors, blend_config=blend_config, device=device, visual_ranges=visual_ranges
            )
        else:
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


def load_vlm_image_tensor(path_str):
    if not path_str:
        return None
    normalized_path = path_str.strip().replace('\\', '/')
    normalized_path = os.path.normpath(normalized_path)
    if not os.path.isabs(normalized_path):
        normalized_path = os.path.abspath(normalized_path)

    if not os.path.isfile(normalized_path):
        raise FileNotFoundError(f"Invalid image path: {path_str} (resolved to: {normalized_path})")

    from PIL import Image, ImageOps, ImageSequence
    import numpy as np

    img = node_helpers.pillow(Image.open, normalized_path)
    for i in ImageSequence.Iterator(img):
        i = node_helpers.pillow(ImageOps.exif_transpose, i)
        if i.mode == 'I':
            i = i.point(lambda x: x * (1 / 65535))
        image = i.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image_np)[None,]  # Returns [1, H, W, C]


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
                io.String.Input("image_paths", multiline=True, default="", placeholder="C:/paths/to/image1.png\nC:/paths/to/image2.png", tooltip="Line-separated list of paths to image files. Must map 1-to-1 with file_names."),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path).",
                ),
                io.String.Input("file_names", multiline=True, default="", placeholder="bulbasaur\nivysaur", tooltip="Line-separated list of file names to save as (without .safetensors). Can include nested subfolders. Must map 1-to-1 with image_paths."),
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
    def execute(cls, clip, prompt, image_paths, vlm_resolution, file_names, slice_visual_tokens=False) -> io.NodeOutput:
        # 1. Parse image paths and file names
        img_paths_list = [p.strip() for p in image_paths.split("\n") if p.strip()]
        file_names_list = [n.strip() for n in file_names.split("\n") if n.strip()]

        if not img_paths_list and not file_names_list:
            raise ValueError("Both image_paths and file_names are empty.")

        # If only text prompt encoding is desired (no images)
        if not img_paths_list:
            if not file_names_list:
                raise ValueError("No file_names specified to save the text embedding.")
            img_paths_list = [None] * len(file_names_list)
        elif not file_names_list:
            raise ValueError("No file_names specified for the provided image paths.")

        if len(img_paths_list) != len(file_names_list):
            raise ValueError(f"Count mismatch: Got {len(img_paths_list)} image paths and {len(file_names_list)} file names.")

        # 2. Pre-Execution Path Validation
        for path in img_paths_list:
            if path is not None:
                normalized_path = path.strip().replace('\\', '/')
                normalized_path = os.path.normpath(normalized_path)
                if not os.path.isabs(normalized_path):
                    normalized_path = os.path.abspath(normalized_path)
                if not os.path.isfile(normalized_path):
                    raise FileNotFoundError(
                        f"Validation aborted: Image file does not exist: '{path}' (resolved to: '{normalized_path}'). "
                        "No processing has started, ensuring safe memory state."
                    )

        # 3. Call clip.load_model() once to register the model as active for comfy-aimdo
        clip.load_model()
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

        last_state_dict = None
        last_tensor_2d = None

        # 4. Process loop under inference_mode
        for img_path, f_name in zip(img_paths_list, file_names_list):
            # Load and preprocess image if present
            images_vl = []
            if img_path is not None:
                image_tensor = load_vlm_image_tensor(img_path)

                # Image resolution downscaling helper
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

                processed_img = process_vlm_image(image_tensor, vlm_resolution)
                images_vl.append(processed_img)

            # Tokenize prompt using skip_template=True so no template wrapping is saved
            modified_prompt = prompt
            if img_path is not None:
                if not any(tag in prompt for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>"]):
                    modified_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + modified_prompt

            tokens = clip.tokenize(modified_prompt, images=images_vl, skip_template=True)

            key_name = next(iter(tokens.keys()))
            token_list = tokens[key_name]
            tokens_only = [[t[0] for t in b] for b in token_list]

            import comfy
            device = comfy.model_management.get_torch_device()

            with torch.inference_mode():
                embeds, _, _, embeds_info = clip_model.process_tokens(tokens_only, device)

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

                tensor_2d = embeds_sliced.squeeze(0).clone().cpu()

            state_dict = {key_name: tensor_2d}

            # Save the safetensors file (ensure nested directories exist)
            target_path = os.path.join(embeddings_dir, f"{f_name}.safetensors")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            state_dict_safe = {k: v.contiguous() for k, v in state_dict.items()}
            save_file(state_dict_safe, target_path)

            last_state_dict = state_dict
            last_tensor_2d = tensor_2d

            # Clean VRAM loop references
            del embeds, embeds_sliced, tokens, tokens_only
            if img_path is not None:
                del image_tensor, processed_img

        # 5. Final VRAM release and soft_empty_cache
        import gc
        gc.collect()
        comfy.model_management.soft_empty_cache()

        return io.NodeOutput(last_state_dict, last_tensor_2d)


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
                io.String.Input("image_paths", multiline=True, default="", placeholder="C:/paths/to/image1.png\nC:/paths/to/image2.png", tooltip="Line-separated list of paths to image files. Must map 1-to-1 with file_names."),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path).",
                ),
                io.String.Input("file_names", multiline=True, default="", placeholder="bulbasaur\nivysaur", tooltip="Line-separated list of file names to save as (without .safetensors). Can include nested subfolders. Must map 1-to-1 with image_paths."),
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
    def execute(cls, clip, prompt, image_paths, vlm_resolution, file_names, slice_visual_tokens=False) -> io.NodeOutput:
        # 1. Parse image paths and file names
        img_paths_list = [p.strip() for p in image_paths.split("\n") if p.strip()]
        file_names_list = [n.strip() for n in file_names.split("\n") if n.strip()]

        if not img_paths_list and not file_names_list:
            raise ValueError("Both image_paths and file_names are empty.")

        # If only text prompt encoding is desired (no images)
        if not img_paths_list:
            if not file_names_list:
                raise ValueError("No file_names specified to save the text embedding.")
            img_paths_list = [None] * len(file_names_list)
        elif not file_names_list:
            raise ValueError("No file_names specified for the provided image paths.")

        if len(img_paths_list) != len(file_names_list):
            raise ValueError(f"Count mismatch: Got {len(img_paths_list)} image paths and {len(file_names_list)} file names.")

        # 2. Pre-Execution Path Validation
        for path in img_paths_list:
            if path is not None:
                normalized_path = path.strip().replace('\\', '/')
                normalized_path = os.path.normpath(normalized_path)
                if not os.path.isabs(normalized_path):
                    normalized_path = os.path.abspath(normalized_path)
                if not os.path.isfile(normalized_path):
                    raise FileNotFoundError(
                        f"Validation aborted: Image file does not exist: '{path}' (resolved to: '{normalized_path}'). "
                        "No processing has started, ensuring safe memory state."
                    )

        # 3. Call clip.load_model() once to register the model as active for comfy-aimdo
        clip.load_model()
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

        last_state_dict = None
        last_tensor_2d = None

        # 4. Process loop under inference_mode
        for img_path, f_name in zip(img_paths_list, file_names_list):
            # Load and preprocess image if present
            images_vl = []
            if img_path is not None:
                image_tensor = load_vlm_image_tensor(img_path)

                # Image resolution downscaling helper
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

                processed_img = process_vlm_image(image_tensor, vlm_resolution)
                images_vl.append(processed_img)

            # Tokenize prompt using skip_template=True so no template wrapping is saved
            modified_prompt = prompt
            if img_path is not None:
                if not any(tag in prompt for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>"]):
                    modified_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + modified_prompt

            tokens = clip.tokenize(modified_prompt, images=images_vl, skip_template=True)

            # Retrieve the key name dynamically (typically "qwen3vl_4b" or "qwen3vl_8b")
            key_name = "qwen3vl_8b"
            if tokens:
                key_name = next(iter(tokens.keys()))
            token_list = tokens.get(key_name, [])
            tokens_only = [[t[0] for t in b] for b in token_list]

            import comfy
            device = comfy.model_management.get_torch_device()

            with torch.inference_mode():
                embeds, _, _, embeds_info = clip_model.process_tokens(tokens_only, device)

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

                tensor_2d = embeds_sliced.squeeze(0).clone().cpu()

            state_dict = {key_name: tensor_2d}

            # Save the safetensors file (ensure nested directories exist)
            target_path = os.path.join(embeddings_dir, f"{f_name}.safetensors")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            state_dict_safe = {k: v.contiguous() for k, v in state_dict.items()}
            save_file(state_dict_safe, target_path)

            last_state_dict = state_dict
            last_tensor_2d = tensor_2d

            # Clean VRAM loop references
            del embeds, embeds_sliced, tokens, tokens_only
            if img_path is not None:
                del image_tensor, processed_img

        # 5. Final VRAM release and soft_empty_cache
        import gc
        gc.collect()
        comfy.model_management.soft_empty_cache()

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
                    tooltip="Main text prompt.",
                ),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.String.Input(
                    "attention_weights",
                    multiline=False,
                    default="",
                    tooltip="Space-separated list of weighted words/phrases. Example: (arms:1.5) (painting:-1) (photo:2)",
                ),
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
                BlendConfig.Input("blend_config", optional=True, tooltip="Optional Consensus Blend Configuration input."),
                io.Autogrow.Input("image_inputs", template=autogrow_template),
            ],
            outputs=[
                io.Model.Output(),
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, clip, prompt, system_prompt, attention_weights, vlm_resolution, multiplier, strength, padding_method, formula, image_inputs: io.Autogrow.Type, blend_config: dict = None) -> io.NodeOutput:
        active_images = []
        if image_inputs is not None:
            for k, v in image_inputs.items():
                if v is not None:
                    active_images.append(v)

        # 1. Parse weights from the attention_weights widget using regex
        import re
        pattern = re.compile(r"\(([^():]+):(-?\d*\.?\d+)\)")
        terms = [(m.group(1).strip(), float(m.group(2))) for m in pattern.finditer(attention_weights)]

        # Prompt inputs remain as untouched plain-text strings
        clean_prompt = prompt
        clean_system_prompt = system_prompt

        if blend_config is None:
            blend_config = {"blend_preset": "off"}
        blend_preset = blend_config.get("blend_preset", "off")

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

            if len(clean_system_prompt) > 0:
                clean_full_prompt = (
                    "<|im_start|>user\n" + "<|im_end|>\n" +
                    "<|im_start|>system\n" + clean_system_prompt + "<|im_end|>\n" +
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
            if len(clean_system_prompt) > 0:
                clean_full_prompt = (
                    "<|im_start|>user\n" + "<|im_end|>\n" +
                    "<|im_start|>system\n" + clean_system_prompt + "<|im_end|>\n" +
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
                matches = _find_subsequence(ids, sub, 0, len(ids))
                if matches:
                    for mi in matches:
                        for off in range(len(sub)):
                            t_idx = mi + off
                            if t_idx < len(mapping):
                                positions.append(mapping[t_idx][0])
                    break
            if not positions:
                import logging
                logging.warning(f"Krea2PromptWeight: phrase '{phrase}' not found in prompt or system prompt; skipped.")
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
            visual_ranges = {}
            ref_cond_dict = None
            last_cond_dict = None

            for idx, img in enumerate(active_images):
                letter = chr(97 + idx)
                processed_img = process_vlm_image(img, vlm_resolution)

                modified_prompt = clean_prompt
                if not any(tag in clean_prompt for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>", "image_input_"]):
                    modified_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + modified_prompt

                if len(clean_system_prompt) > 0:
                    full_prompt = (
                        "<|im_start|>user\n" + "<|im_end|>\n" +
                        "<|im_start|>system\n" + clean_system_prompt + "<|im_end|>\n" +
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

                # Tokenize and find visual token range for isolation blending
                try:
                    tokens = clip.tokenize(full_prompt, images=[processed_img], skip_template=True)
                    vis_start, vis_end = find_visual_token_range(tokens, C_X)
                    visual_ranges[letter] = (vis_start, vis_end)
                except Exception:
                    visual_ranges[letter] = (0, 0)

                if "embeds_info" in cond_X[0][1] and len(cond_X[0][1]["embeds_info"]) > 0:
                    extra = cond_X[0][1]["embeds_info"][0].get("extra", {})
                    if "deepstack" in extra:
                        deepstack_dict[letter] = extra["deepstack"]

                if idx == 0:
                    ref_cond_dict = cond_X[0][1]
                last_cond_dict = cond_X[0][1]

            if blend_preset != "off":
                import comfy
                device = comfy.model_management.get_torch_device()
                C_blended, P_blended = evaluate_conditioning_consensus_blend(
                    sequence_tensors, pooled_tensors, blend_config=blend_config, device=device, visual_ranges=visual_ranges
                )
            else:
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

                    if blend_preset != "off":
                        wrapped_layer_tensors = {let: t.unsqueeze(0) for let, t in layer_tensors.items()}
                        import comfy
                        device = comfy.model_management.get_torch_device()
                        # DeepStack intermediate layers must be blended purely using similarity weights
                        # and must NEVER carry global scales or vector magnitude norm rescalings which distort inner residuals.
                        ds_blend_config = blend_config.copy()
                        ds_blend_config["global_scale"] = 1.0
                        ds_blend_config["rescale_norm"] = False
                        ds_blend_config["alignment_method"] = "index"
                        C_l_blended, _ = evaluate_conditioning_consensus_blend(
                            wrapped_layer_tensors, {}, blend_config=ds_blend_config, device=device
                        )
                        deepstack_blended.append(C_l_blended.squeeze(0))
                    else:
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

            conditioning = [[C_blended, final_cond_dict]]
        else:
            conditioning = encode_embedding_classical_scaled_bias(clip, clean_full_prompt)
            if multiplier != 1.0:
                for i in range(len(conditioning)):
                    conditioning[i][0] *= multiplier
                    if "pooled_output" in conditioning[i][1] and conditioning[i][1]["pooled_output"] is not None:
                        conditioning[i][1]["pooled_output"] *= multiplier

        return io.NodeOutput(model_clone, conditioning)


