import os
import math
import re
import torch
import logging
from einops import rearrange
from safetensors.torch import save_file
from enum import Enum
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageSequence
import numpy as np

import folder_paths
import node_helpers
import comfy

from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention, attention_pytorch


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
                tensor_perm = tensor.permute(0, 2, 1)
                tensor_interp = F.interpolate(tensor_perm, size=max_len, mode='linear', align_corners=False)
                tensor = tensor_interp.permute(0, 2, 1)
            else: # zero-pad
                pad_size = max_len - tensor.shape[1]
                padding = torch.zeros((tensor.shape[0], pad_size, tensor.shape[2]), device=tensor.device, dtype=tensor.dtype)
                tensor = torch.cat([tensor, padding], dim=1)
        # Cast the padded tensor to the target device via Comfy's non-blocking, aimdo-aware pipeline
        aligned_sequence_tensors[name] = comfy.model_management.cast_to_device(tensor, tensor.device, tensor.dtype)

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

def reconstruct_2d_grid(N: int) -> tuple:
    """
    Determines the closest 2D grid dimensions (H, W) for N tokens.
    """
    root = int(math.sqrt(N))
    if root * root == N:
        return root, root
    for w in range(root, 0, -1):
        if N % w == 0:
            return N // w, w
    return N, 1

def generate_spatial_fusion_mask(N: int, num_sources: int, method: str, block_size: int = 2, dither_ratio: float = 0.5, device: str = "cpu") -> torch.Tensor:
    """
    Generates a deterministic 1D token index mapping array corresponding to a source image index.
    """
    if num_sources <= 1:
        return torch.zeros(N, dtype=torch.long, device=device)

    h, w = reconstruct_2d_grid(N)

    if method == "spatial-checkerboard":
        mask = torch.zeros(N, dtype=torch.long, device=device)
        for i in range(N):
            r = i // w
            c = i % w
            mask[i] = (r + c) % num_sources
        return mask

    elif method == "spatial-block-interleave":
        mask = torch.zeros(N, dtype=torch.long, device=device)
        for i in range(N):
            r = i // w
            c = i % w
            block_r = r // block_size
            block_c = c // block_size
            mask[i] = (block_r + block_c) % num_sources
        return mask

    elif method == "spatial-dither-random":
        g = torch.Generator(device=device)
        g.manual_seed(42)  # Deterministic seed to prevent shifting patterns on every generation run
        rands = torch.rand(N, generator=g, device=device)
        if num_sources == 2:
            return torch.where(rands < dither_ratio, 0, 1)
        else:
            return (rands * num_sources).long()

    else:
        return torch.zeros(N, dtype=torch.long, device=device)


# =====================================================================
# SYSTEM WARNING FOR FUTURE AGENTS / DEVELOPERS:
# This helper function `save_blended_visual_embeddings` is EXCLUSIVELY
# for saving isolated, spatially-fused visual token embeddings
# generated during VLM image component blending.
# It contains ZERO text prompt tokens, prefixes, or suffixes.
# It MUST serialize the tensor under the model's active embedding_key
# (e.g. 'qwen3vl_8b') in pure torch.float32 on the CPU.
# =====================================================================
def save_blended_visual_embeddings(
    blended_vis_all_batches: list,
    visual_fusion_config: dict,
    embedding_key: str = "qwen3vl_8b"
) -> None:
    """
    Saves isolated, mathematically-fused visual token embeddings as a
    standalone .safetensors file under ComfyUI's models/embeddings directory.

    Warning: This function does NOT save any surrounding prompt or text tokens.
    It expects a list of 2D visual token tensors, stacks them into a batch
    dimension, squeezes it if there is only one batch slice, and writes the
    contiguous tensor under the active VLM tokenizer's embedding_key.

    Parameters:
        blended_vis_all_batches: List of 2D torch.Tensor representing the
                                 visual tokens across the execution batches.
        visual_fusion_config: Dictionary containing target path options.
        embedding_key: The string identifier (e.g., 'qwen3vl_8b' or 'krea2_vlm')
                       required by the model checkpoint reader.
    """

    if not blended_vis_all_batches:
        raise ValueError("[save_blended_visual_embeddings] Cannot save empty visual token list.")

    # Stack batches into [B, max_vis_len, D] and move to CPU in float32 precision
    stacked_vis = torch.stack(blended_vis_all_batches, dim=0).to(device="cpu", dtype=torch.float32)

    # Squeeze batch dimension if B == 1 to keep it as a standard 2D embedding tensor
    if stacked_vis.shape[0] == 1:
        stacked_vis = stacked_vis.squeeze(0)

    save_name = visual_fusion_config.get("save_path", "blended_visual_embeds").strip()
    if not save_name.endswith(".safetensors"):
        save_name += ".safetensors"

    try:
        embed_paths = folder_paths.get_folder_paths("embeddings")
        embeddings_dir = embed_paths[0] if embed_paths else "models/embeddings"
    except Exception:
        embeddings_dir = "models/embeddings"

    os.makedirs(embeddings_dir, exist_ok=True)
    full_save_path = os.path.join(embeddings_dir, save_name)

    # State dict must contain exactly one layer matching the VLM's dynamic embedding_key
    state_dict = {embedding_key: stacked_vis.contiguous()}
    save_file(state_dict, full_save_path)
    logging.info(f"[UC_VisualFusionConfig] Saved blended visual tokens as {embedding_key} embedding to: {full_save_path}")


def evaluate_conditioning_consensus_blend(
    sequence_tensors: dict,
    pooled_tensors: dict,
    visual_fusion_config: dict = None,
    device: str = "cpu",
    visual_ranges: dict = None,
    embedding_key: str = "qwen3vl_8b",
    clip = None,
    tokens_dict: dict = None
) -> tuple:
    """
    Decoupled blending engine focused entirely on isolated visual token spatial fusion.
    Saves isolated blended raw visual embeddings if save_blended_embeds is configured.
    """

    if visual_fusion_config is None:
        visual_fusion_config = {"visual_fusion_method": "spatial-checkerboard", "visual_block_size": 2, "dither_ratio": 0.5}

    visual_method = visual_fusion_config.get("visual_fusion_method", "spatial-checkerboard")

    active_keys = sorted(list(sequence_tensors.keys()))
    tensors_list = [sequence_tensors[k] for k in active_keys]
    if not tensors_list:
        return None, None

    B = tensors_list[0].shape[0]
    D = tensors_list[0].shape[2]

    # --- Independent Standalone RAW Visual Embeddings Extraction & Saving ---
    if visual_fusion_config.get("save_blended_embeds", False) and clip is not None and tokens_dict:
        try:
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

            raw_visuals = {}
            for k in active_keys:
                if k in tokens_dict:
                    tok = tokens_dict[k]
                    key_name = next(iter(tok.keys()))
                    token_list = tok[key_name]
                    tokens_only = [[t[0] for t in b] for b in token_list]
                    with torch.inference_mode():
                        embeds, _, _, embeds_info = clip_model.process_tokens(tokens_only, device)

                    vr = visual_ranges.get(k, (0, 0))
                    if vr and vr != (0, 0):
                        v_start, v_end = vr
                        raw_visuals[k] = embeds[0, v_start:v_end, :].clone().cpu()

            if raw_visuals:
                max_vis_len_raw = max(v.shape[0] for v in raw_visuals.values())
                D_raw = next(iter(raw_visuals.values())).shape[1]

                aligned_raw = {}
                for k, v in raw_visuals.items():
                    if v.shape[0] != max_vis_len_raw:
                        v_perm = v.permute(1, 0).unsqueeze(0)
                        v_interp = torch.nn.functional.interpolate(v_perm, size=max_vis_len_raw, mode='linear', align_corners=False)
                        v = v_interp.squeeze(0).permute(1, 0)
                    aligned_raw[k] = v

                if visual_method.startswith("spatial-"):
                    blended_raw_2d = torch.zeros((max_vis_len_raw, D_raw), device="cpu", dtype=torch.float32)
                    sources_list = [aligned_raw[k] for k in active_keys if k in aligned_raw]

                    fusion_mask = generate_spatial_fusion_mask(
                        N=max_vis_len_raw,
                        num_sources=len(sources_list),
                        method=visual_method,
                        block_size=visual_fusion_config.get("visual_block_size", 2),
                        dither_ratio=visual_fusion_config.get("dither_ratio", 0.5),
                        device="cpu"
                    )

                    for i in range(max_vis_len_raw):
                        src_idx = fusion_mask[i].item()
                        blended_raw_2d[i] = sources_list[src_idx][i]
                elif visual_method == "linear":
                    sources_list = [aligned_raw[k] for k in active_keys if k in aligned_raw]
                    stacked = torch.stack(sources_list, dim=0)
                    blended_raw_2d = torch.mean(stacked, dim=0)
                else:
                    blended_raw_2d = aligned_raw[active_keys[0]]

                # Save correct raw visual input embeddings [N, D_raw] (e.g. 2560) directly
                save_blended_visual_embeddings([blended_raw_2d], visual_fusion_config, embedding_key)
        except Exception as e:
            logging.error(f"[evaluate_conditioning_consensus_blend] Failed to process and save raw embeddings: {e}", exc_info=True)

    C_blended_list = []
    for b in range(B):
        batch_tensors_dict = {k: sequence_tensors[k][b].to(device=device, dtype=torch.float32) for k in active_keys}
        ref_key = active_keys[0]

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

        # Splicing of isolated visual blocks
        active_visuals = {k: v for k, v in visuals.items() if v.shape[0] > 0}
        if active_visuals and visual_method != "off":
            max_vis_len = max(v.shape[0] for v in active_visuals.values())
            aligned_visuals = {}
            for k, v in active_visuals.items():
                if v.shape[0] != max_vis_len:
                    v_perm = v.permute(1, 0).unsqueeze(0)
                    v_interp = torch.nn.functional.interpolate(v_perm, size=max_vis_len, mode='linear', align_corners=False)
                    v = v_interp.squeeze(0).permute(1, 0)
                aligned_visuals[k] = v

            if visual_method.startswith("spatial-"):
                blended_vis_2d = torch.zeros((max_vis_len, D), device=device, dtype=torch.float32)
                sources_list = [aligned_visuals[k] for k in active_keys if k in aligned_visuals]

                fusion_mask = generate_spatial_fusion_mask(
                    N=max_vis_len,
                    num_sources=len(sources_list),
                    method=visual_method,
                    block_size=visual_fusion_config.get("visual_block_size", 2),
                    dither_ratio=visual_fusion_config.get("dither_ratio", 0.5),
                    device=device
                )

                for i in range(max_vis_len):
                    src_idx = fusion_mask[i].item()
                    blended_vis_2d[i] = sources_list[src_idx][i]

            elif visual_method == "linear":
                # Fallback linear average
                sources_list = [aligned_visuals[k] for k in active_keys if k in aligned_visuals]
                stacked = torch.stack(sources_list, dim=0)
                blended_vis_2d = torch.mean(stacked, dim=0)
            else:
                # Default fallback to reference image
                blended_vis_2d = aligned_visuals[ref_key]
        else:
            blended_vis_2d = aligned_visuals[ref_key] if active_visuals else torch.zeros((0, D), device=device)

        # Surrounding text (prefixes & suffixes) are kept 100% pure from the reference pass
        blended_prefix = prefixes[ref_key]
        blended_suffix = suffixes[ref_key]

        # Stitch segments back together
        C_blended_list.append(torch.cat([blended_prefix, blended_vis_2d, blended_suffix], dim=0))

    C_blended = torch.stack(C_blended_list, dim=0).to(dtype=tensors_list[0].dtype, device=tensors_list[0].device)

    # Pooled output is kept pure from reference pass since text is identical
    ref_key = active_keys[0]
    P_blended = pooled_tensors.get(ref_key, None)

    return C_blended, P_blended


def blend_text_vectors(sequence_tensors: dict, blend_config: dict, pooled_tensors: dict = None, device: str = "cpu") -> tuple:
    """
    Consensus-Weighted Blending math engine for language space sequences and pooled embeddings.
    Used ONLY post-encoder inside UC_ConditioningConsensusBlend.
    """
    active_keys = sorted(list(sequence_tensors.keys()))
    tensors_list = [sequence_tensors[k] for k in active_keys]

    B = tensors_list[0].shape[0]
    D = tensors_list[0].shape[2]

    blend_preset = blend_config.get("blend_preset", "baseline")
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
        "baseline": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.0, "thresh": 0.0, "beta": 0.0, "scale": 1.0, "norm": False, "dsc": False, "soft_comfort": False},
        "high_clarity": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 3.0, "thresh": 0.3, "beta": 0.0, "scale": 1.0, "norm": False, "dsc": False, "soft_comfort": False},
        "smooth": {"method": "consensus", "type": "mean", "align": "similarity", "alpha": 1.5, "thresh": 0.0, "beta": 0.0, "scale": 1.0, "norm": False, "dsc": False, "soft_comfort": False},
        "varied_merge": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.0, "thresh": 0.0, "beta": 0.0, "scale": 0.7, "norm": True, "dsc": False, "soft_comfort": False},
        "diverse_concept": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.0, "thresh": 0.0, "beta": 1.0, "scale": 0.7, "norm": True, "dsc": False, "soft_comfort": False},
        "high_diversity_concept": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.0, "thresh": 0.0, "beta": 2.0, "scale": 0.7, "norm": True, "dsc": False, "soft_comfort": False},
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

                if dsc_enabled:
                    min_sim = similarities.min()
                    max_sim = similarities.max()
                    if max_sim > min_sim:
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
            # Index-Based Sequential Matching
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

                row_weights = torch.zeros_like(similarities)
                mask = similarities >= similarity_threshold

                if mask.any():
                    if diversity_beta > 0.0:
                        row_weights[mask] = torch.pow(similarities[mask], power_alpha) * torch.pow(similarities[mask], diversity_beta)
                    else:
                        row_weights[mask] = torch.pow(similarities[mask], power_alpha)
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

    # Blend metadata pooled outputs
    P_blended = None
    if pooled_tensors and any(p is not None for p in pooled_tensors.values()):
        pooled_list_active = [pooled_tensors[k] for k in active_keys if pooled_tensors.get(k) is not None]
        stacked_pooled = torch.stack(pooled_list_active, dim=1)
        P_blended_batches = []
        for b in range(B):
            stacked_p = stacked_pooled[b]
            if consensus_type == "median":
                consensus_p = torch.median(stacked_p, dim=0).values
            else:
                consensus_p = torch.mean(stacked_p, dim=0)

            stacked_p_norm = torch.nn.functional.normalize(stacked_p, p=2, dim=1)
            consensus_p_norm = torch.nn.functional.normalize(consensus_p, p=2, dim=0)
            similarities_p = torch.mv(stacked_p_norm, consensus_p_norm)

            weights_p = torch.zeros_like(similarities_p)
            mask_p = similarities_p >= similarity_threshold

            if mask_p.any():
                if diversity_beta > 0.0:
                    weights_p[mask_p] = torch.pow(similarities_p[mask_p], power_alpha) * torch.pow(1.001 - similarities_p[mask_p], diversity_beta)
                else:
                    weights_p[mask_p] = torch.pow(similarities_p[mask_p], power_alpha)
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

def load_vlm_image_tensor(path_str):
    if not path_str:
        return None
    normalized_path = path_str.strip().replace('\\', '/')
    normalized_path = os.path.normpath(normalized_path)
    if not os.path.isabs(normalized_path):
        normalized_path = os.path.abspath(normalized_path)

    if not os.path.isfile(normalized_path):
        raise FileNotFoundError(f"Invalid image path: {path_str} (resolved to: {normalized_path})")


    img = node_helpers.pillow(Image.open, normalized_path)
    for i in ImageSequence.Iterator(img):
        i = node_helpers.pillow(ImageOps.exif_transpose, i)
        if i.mode == 'I':
            i = i.point(lambda x: x * (1 / 65535))
        image = i.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image_np)[None,]  # Returns [1, H, W, C]

def krea2_user_content_span(ids):
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

def krea2_token_ids(clip, text):
    tok = clip.tokenize(text)
    key = next(iter(tok))
    return [t[0] if isinstance(t, tuple) else t for t in tok[key][0]]

def find_subsequence(seq, sub, lo, hi):
    out = []
    n = len(sub)
    if n == 0:
        return out
    for i in range(lo, hi - n + 1):
        if seq[i:i + n] == sub:
            out.append(i)
    return out


def krea2_attn_forward_weight(self, x, freqs=None, mask=None, transformer_options={}):

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

