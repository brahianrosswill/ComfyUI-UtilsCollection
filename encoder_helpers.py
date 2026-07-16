import ast
import operator
import os
import math
import re
import torch
import logging
import numbers
from einops import rearrange
from safetensors.torch import save_file
from enum import Enum
from pathlib import Path
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageSequence
import numpy as np

import folder_paths
import node_helpers
import comfy

from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention
from comfy.sd1_clip import token_weights
from comfy.text_encoders.qwen_vl import qwen2vl_image_size


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

    # Apply contextual vector scaling directly to each schedule. Pooled output is
    # deliberately unchanged: local token weights do not define a pooled weight.
    new_conditioning = []

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

        new_conditioning.append([new_cond, cond_dict.copy()])

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


_IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\bimage_input_(fusion|\d+)\b", re.IGNORECASE)
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


def prepare_image_placeholder_prompt(prompt: str, image_count: int, fusion_active: bool, context: str) -> tuple[str, tuple[int, ...]]:
    """Normalize custom image placeholders without leaving invalid names as text."""
    matches = list(_IMAGE_PLACEHOLDER_PATTERN.finditer(prompt))

    if fusion_active:
        if any(tag in prompt for tag in ("<|image_pad|>", "<|image|>", "<|vision_start|>")):
            if matches:
                logging.warning(
                    "%s: native visual tokens already exist; stripped %d image_input placeholder(s).",
                    context,
                    len(matches),
                )
            return _IMAGE_PLACEHOLDER_PATTERN.sub("", prompt), ()

        chosen = next((match for match in matches if match.group(1).lower() == "fusion"), None)
        if chosen is None:
            chosen = next((match for match in matches if match.group(1) == "1"), None)

        if chosen is None:
            if matches:
                logging.warning(
                    "%s: fusion accepts image_input_fusion or image_input_1; stripped %d unsupported placeholder(s).",
                    context,
                    len(matches),
                )
            logging.warning("%s: no fusion placeholder found; prepended the fused visual slot.", context)
            return VISION_BLOCK + _IMAGE_PLACEHOLDER_PATTERN.sub("", prompt), ()

        if chosen.group(1).lower() == "1":
            logging.warning("%s: treating image_input_1 as image_input_fusion.", context)
        if len(matches) > 1:
            logging.warning(
                "%s: fusion uses one visual slot; stripped %d additional image_input placeholder(s).",
                context,
                len(matches) - 1,
            )

        def replace_fusion(match):
            return VISION_BLOCK if match.start() == chosen.start() else ""

        return _IMAGE_PLACEHOLDER_PATTERN.sub(replace_fusion, prompt), ()

    valid_numbers = []
    removed = []

    def validate_numbered(match):
        suffix = match.group(1).lower()
        if suffix == "fusion":
            removed.append(match.group(0))
            return ""
        number = int(suffix)
        if number < 1 or number > image_count:
            removed.append(match.group(0))
            return ""
        valid_numbers.append(number)
        return VISION_BLOCK

    rewritten = _IMAGE_PLACEHOLDER_PATTERN.sub(validate_numbered, prompt)
    if removed:
        logging.warning(
            "%s: stripped unavailable or fusion-only placeholder(s): %s.",
            context,
            ", ".join(removed),
        )
    return rewritten, tuple(valid_numbers)

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


_QWEN_IM_START, _QWEN_USER, _QWEN_NL, _QWEN_IM_END = 151644, 872, 198, 151645


def _token_id(token):
    value = token[0] if isinstance(token, tuple) and token else token
    return int(value) if isinstance(value, numbers.Integral) else None


def _krea2_prefix_candidates(token_list) -> list[int]:
    """Return the released and pending Krea2 prefix-strip boundaries."""
    seen = 0
    ids = [_token_id(token) for token in token_list]
    released_end = -1
    for index, token_id in enumerate(ids):
        if token_id == _QWEN_IM_START and seen < 2:
            released_end = index
            seen += 1
    if released_end >= 0 and ids[released_end + 1:released_end + 3] == [_QWEN_USER, _QWEN_NL]:
        released_end += 3

    image_position = next((index for index, token in enumerate(token_list) if is_image_token(token)), len(token_list))
    pending_end = -1
    for index in range(max(0, image_position - 2)):
        if ids[index:index + 3] == [_QWEN_IM_START, _QWEN_USER, _QWEN_NL]:
            pending_end = index + 3

    return sorted({candidate for candidate in (released_end, pending_end) if candidate >= 0})


def _qwen3vl_image_span(token) -> int | None:
    value = token[0] if isinstance(token, tuple) and token else token
    if not isinstance(value, dict) or value.get("type") != "image":
        return None
    image = value.get("data")
    if not torch.is_tensor(image) or image.ndim != 4:
        return None
    height, width = image.shape[1:3]
    resized_height, resized_width = qwen2vl_image_size(
        height,
        width,
        min_pixels=3136,
        max_pixels=12845056,
        patch_size=16,
        merge_size=2,
    )
    return (resized_height // 16) * (resized_width // 16) // 4


def build_token_to_conditioning_map(token_list, cond_tensor) -> list[tuple[int, int]]:
    """Map raw tokenizer entries to conditioning spans, validating all inferred lengths."""
    cond_len = cond_tensor.shape[1]
    is_krea2 = cond_tensor.shape[-1] == 12 * 2560
    prefix_candidates = _krea2_prefix_candidates(token_list) if is_krea2 else [0]
    if not prefix_candidates:
        raise ValueError("Could not locate a supported Krea2 user-prompt prefix boundary.")

    prefix_len = None
    token_spans = None
    candidate_details = []
    for candidate in prefix_candidates:
        retained = token_list[candidate:]
        exact_spans = [_qwen3vl_image_span(token) if is_image_token(token) else 1 for token in retained]
        if all(span is not None for span in exact_spans):
            mapped_length = sum(exact_spans)
            candidate_details.append((candidate, mapped_length, "exact-grid"))
            if mapped_length == cond_len:
                prefix_len = candidate
                token_spans = exact_spans
                break
            continue

        image_count = sum(is_image_token(token) for token in retained)
        text_count = len(retained) - image_count
        if image_count == 0:
            candidate_details.append((candidate, text_count, "text-only"))
            if text_count == cond_len:
                prefix_len = candidate
                token_spans = [1] * len(retained)
                break
        else:
            expanded = cond_len - text_count
            candidate_details.append((candidate, expanded, "uniform-fallback"))
            if expanded >= image_count and expanded % image_count == 0:
                image_span = expanded // image_count
                prefix_len = candidate
                token_spans = [image_span if is_image_token(token) else 1 for token in retained]
                break

    if prefix_len is None:
        raise ValueError(
            "No supported tokenizer-prefix contract maps exactly to the returned conditioning length "
            f"(conditioning_length={cond_len}, candidates={candidate_details})."
        )

    mapping = []
    current = 0
    retained_index = 0
    for index, token in enumerate(token_list):
        if index < prefix_len:
            mapping.append((-1, -1))
            continue
        size = token_spans[retained_index]
        retained_index += 1
        mapping.append((current, current + size))
        current += size
    if current != cond_len:
        raise ValueError(f"Token mapping ended at {current}, expected conditioning length {cond_len}.")
    return mapping

_FORMULA_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_FORMULA_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _formula_min(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not torch.is_tensor(a):
            a = torch.as_tensor(a, device=b.device, dtype=b.dtype)
        if not torch.is_tensor(b):
            b = torch.as_tensor(b, device=a.device, dtype=a.dtype)
        return torch.minimum(a, b)
    return min(a, b)


def _formula_max(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not torch.is_tensor(a):
            a = torch.as_tensor(a, device=b.device, dtype=b.dtype)
        if not torch.is_tensor(b):
            b = torch.as_tensor(b, device=a.device, dtype=a.dtype)
        return torch.maximum(a, b)
    return max(a, b)


_FORMULA_FUNCTIONS = {
    "abs": abs,
    "min": _formula_min,
    "max": _formula_max,
    "clamp": torch.clamp,
}


def evaluate_tensor_expression(expression: str, variables: dict):
    """Evaluate the documented tensor-expression grammar without Python eval."""
    try:
        root = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc.msg}") from exc

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"Unknown expression variable: {node.id}")
            return variables[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _FORMULA_BINOPS:
            return _FORMULA_BINOPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _FORMULA_UNARYOPS:
            return _FORMULA_UNARYOPS[type(node.op)](visit(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORMULA_FUNCTIONS:
            if node.keywords:
                raise ValueError("Keyword arguments are not supported in expressions.")
            return _FORMULA_FUNCTIONS[node.func.id](*(visit(arg) for arg in node.args))
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    result = visit(root)
    if torch.is_tensor(result) and not torch.isfinite(result).all():
        raise ValueError("Expression produced NaN or infinite values.")
    return result


def evaluate_formula(expression: str, processed_images: dict) -> torch.Tensor:
    try:
        result = evaluate_tensor_expression(expression, processed_images)
        if not torch.is_tensor(result):
            reference = next(iter(processed_images.values()), None)
            if reference is None:
                raise ValueError("A visual formula requires at least one image variable.")
            result = torch.full_like(reference, float(result))
        return torch.clamp(result, 0.0, 1.0)
    except Exception as e:
        raise RuntimeError(f"Error evaluating visual math expression '{expression}': {e}") from e

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

    try:
        C_blended = evaluate_tensor_expression(expression, aligned_sequence_tensors)
        P_blended = None
        if any(v is not None for v in pooled_tensors.values()):
            pooled_variables = {name: tensor for name, tensor in pooled_tensors.items() if tensor is not None}
            missing = set(aligned_sequence_tensors) - set(pooled_variables)
            if missing:
                raise ValueError(f"Formula references conditioning sources without pooled outputs: {sorted(missing)}")
            P_blended = evaluate_tensor_expression(expression, pooled_variables)
        if not torch.is_tensor(C_blended) or C_blended.ndim != 3:
            raise ValueError("Conditioning formula must produce a [batch, tokens, channels] tensor.")
        return C_blended, P_blended
    except Exception as e:
        raise RuntimeError(f"Error evaluating conditioning math expression '{expression}': {e}") from e

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

SPATIAL_FUSION_METHODS = {"spatial-checkerboard", "spatial-block-interleave", "spatial-dither-random"}
VISUAL_FUSION_METHODS = SPATIAL_FUSION_METHODS | {"linear"}


def generate_spatial_fusion_mask(N: int, num_sources: int, method: str, block_size: int = 2, dither_ratio: float = 0.5, device: str = "cpu", seed: int = 0) -> torch.Tensor:
    """
    Generates a seeded 1D token index mapping array corresponding to a source image index.
    """
    if N < 0:
        raise ValueError("Visual token count cannot be negative.")
    if num_sources < 1:
        raise ValueError("Visual fusion requires at least one source.")
    if method not in SPATIAL_FUSION_METHODS:
        raise ValueError(f"Unsupported spatial fusion method: {method}")
    if method == "spatial-block-interleave" and block_size < 1:
        raise ValueError("Visual block size must be at least 1.")
    if method == "spatial-dither-random" and not 0.0 <= dither_ratio <= 1.0:
        raise ValueError("Dither ratio must be between 0.0 and 1.0.")
    if not 0 <= seed <= 0xffffffffffffffff:
        raise ValueError("Visual fusion seed must be between 0 and 18446744073709551615.")
    if num_sources == 1:
        return torch.zeros(N, dtype=torch.long, device=device)

    h, w = reconstruct_2d_grid(N)
    rows = torch.arange(h, device=device).unsqueeze(1)
    columns = torch.arange(w, device=device).unsqueeze(0)

    if method == "spatial-checkerboard":
        return ((rows + columns) % num_sources).flatten()
    if method == "spatial-block-interleave":
        return ((rows // block_size + columns // block_size) % num_sources).flatten()

    generator = torch.Generator(device=device).manual_seed(seed)
    random = torch.rand(N, generator=generator, device=device)
    other_sources = 1 + ((rows + columns) % (num_sources - 1)).flatten()
    return torch.where(random < dither_ratio, 0, other_sources)


def _visual_fusion_mask(config, N, num_sources, mask_device, output_device, mask_cache):
    method = config.get("visual_fusion_method", "spatial-checkerboard")
    block_size = config.get("visual_block_size", 2)
    dither_ratio = config.get("dither_ratio", 0.5)
    seed = config.get("seed", 0)
    key = (N, num_sources, method, block_size, dither_ratio, seed)
    if key not in mask_cache:
        mask_cache[key] = generate_spatial_fusion_mask(N, num_sources, method, block_size, dither_ratio, mask_device, seed)
    return mask_cache[key].to(output_device)


def fuse_visual_token_sources(sources, visual_fusion_config, mask_device, mask_cache=None, expected_length=None):
    if not sources:
        raise ValueError("Visual fusion requires at least one visual token source.")

    method = visual_fusion_config.get("visual_fusion_method", "spatial-checkerboard")
    if method not in VISUAL_FUSION_METHODS:
        raise ValueError(f"Unsupported visual fusion method: {method}")
    if any(source.ndim != 2 for source in sources):
        raise ValueError("Visual fusion sources must have shape [tokens, dimensions].")
    if any(source.shape[1] != sources[0].shape[1] for source in sources[1:]):
        raise ValueError("Visual fusion sources must have matching embedding dimensions.")
    if any(source.device != sources[0].device for source in sources[1:]):
        raise ValueError("Visual fusion sources must be on the same device.")

    max_length = max(source.shape[0] for source in sources)
    if expected_length is not None and max_length != expected_length:
        raise ValueError(f"Visual token layout mismatch: expected {expected_length} tokens, received {max_length}.")

    output_dtype = sources[0].dtype
    compute_float = method == "linear" or any(source.shape[0] != max_length for source in sources)
    aligned = []
    for source in sources:
        value = source.to(dtype=torch.float32) if compute_float else source
        if value.shape[0] != max_length:
            value = F.interpolate(value.transpose(0, 1).unsqueeze(0), size=max_length, mode="linear", align_corners=False).squeeze(0).transpose(0, 1)
        aligned.append(value)

    stacked = torch.stack(aligned, dim=1)
    if method == "linear":
        return stacked.mean(dim=1).to(dtype=output_dtype)

    if mask_cache is None:
        mask_cache = {}
    mask = _visual_fusion_mask(visual_fusion_config, max_length, len(sources), mask_device, stacked.device, mask_cache)
    fused = torch.take_along_dim(stacked, mask[:, None, None], dim=1).squeeze(1)
    return fused.to(dtype=output_dtype)


def fuse_deepstack_layers(deepstack_tensors, visual_fusion_config, device, mask_cache, expected_length):
    active_keys = sorted(deepstack_tensors)
    if not active_keys:
        return None

    num_layers = len(deepstack_tensors[active_keys[0]])
    if any(len(deepstack_tensors[key]) != num_layers for key in active_keys[1:]):
        raise ValueError("Visual fusion sources produced different DeepStack layer counts.")

    blended = []
    for layer in range(num_layers):
        sources = [deepstack_tensors[key][layer].to(device=device) for key in active_keys]
        blended.append(fuse_visual_token_sources(sources, visual_fusion_config, device, mask_cache, expected_length))
    return blended


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

    embed_paths = folder_paths.get_folder_paths("embeddings")
    if not embed_paths:
        raise ValueError("No ComfyUI embeddings directory is configured.")
    embeddings_dir = embed_paths[0]
    full_save_path = resolve_embedding_output_path(embeddings_dir, save_name)
    os.makedirs(os.path.dirname(full_save_path), exist_ok=True)

    # State dict must contain exactly one layer matching the VLM's dynamic embedding_key
    state_dict = {embedding_key: stacked_vis.contiguous()}
    save_file(state_dict, full_save_path)
    logging.info(f"[UC_VisualFusionConfig] Saved blended visual tokens as {embedding_key} embedding to: {full_save_path}")


def resolve_embedding_output_path(embeddings_dir: str, file_name: str) -> str:
    """Resolve a relative output name and prove that it remains below embeddings_dir."""
    if not file_name or not file_name.strip():
        raise ValueError("Embedding file name cannot be empty.")
    relative = Path(file_name.strip())
    if relative.is_absolute():
        raise ValueError("Embedding file name must be relative to the embeddings directory.")
    root = Path(embeddings_dir).resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("Embedding output path escapes the embeddings directory.") from exc
    return str(target)


def evaluate_conditioning_consensus_blend(
    sequence_tensors: dict,
    pooled_tensors: dict,
    visual_fusion_config: dict = None,
    device: str = "cpu",
    visual_ranges: dict = None,
    embedding_key: str = "qwen3vl_8b",
    clip = None,
    tokens_dict: dict = None,
    mask_cache: dict = None,
) -> tuple:
    """
    Decoupled blending engine focused entirely on isolated visual token spatial fusion.
    Saves isolated blended raw visual embeddings if save_blended_embeds is configured.
    """

    if visual_fusion_config is None:
        visual_fusion_config = {"visual_fusion_method": "spatial-checkerboard", "visual_block_size": 2, "dither_ratio": 0.5, "seed": 0}

    visual_method = visual_fusion_config.get("visual_fusion_method", "spatial-checkerboard")
    if visual_method not in VISUAL_FUSION_METHODS:
        raise ValueError(f"Unsupported visual fusion method: {visual_method}")
    if visual_ranges is None:
        raise ValueError("Visual token ranges are required for visual fusion.")
    if mask_cache is None:
        mask_cache = {}

    active_keys = sorted(list(sequence_tensors.keys()))
    tensors_list = [sequence_tensors[k] for k in active_keys]
    if not tensors_list:
        return None, None

    B = tensors_list[0].shape[0]
    expected_visual_length = max(end - start for start, end in (visual_ranges.get(key, (0, 0)) for key in active_keys))
    if expected_visual_length <= 0 or any(visual_ranges.get(key, (0, 0)) == (0, 0) for key in active_keys):
        raise ValueError("Every visual fusion source must have a valid visual token range.")

    C_blended_list = []
    for b in range(B):
        batch_tensors_dict = {k: sequence_tensors[k][b].to(device=device) for k in active_keys}
        ref_key = active_keys[0]

        prefixes = {}
        visuals = {}
        suffixes = {}

        for k in active_keys:
            t = batch_tensors_dict[k]
            v_start, v_end = visual_ranges[k]
            prefixes[k] = t[:v_start, :]
            visuals[k] = t[v_start:v_end, :]
            suffixes[k] = t[v_end:, :]

        sources = [visuals[key] for key in active_keys]
        blended_vis_2d = fuse_visual_token_sources(sources, visual_fusion_config, device, mask_cache, expected_visual_length)

        # Surrounding text (prefixes & suffixes) are kept 100% pure from the reference pass
        blended_prefix = prefixes[ref_key]
        blended_suffix = suffixes[ref_key]

        # Stitch segments back together
        C_blended_list.append(torch.cat([blended_prefix, blended_vis_2d, blended_suffix], dim=0))

    C_blended = torch.stack(C_blended_list, dim=0).to(dtype=tensors_list[0].dtype, device=tensors_list[0].device)

    if visual_fusion_config.get("save_blended_embeds", False):
        if clip is None or not tokens_dict:
            raise ValueError("Saving blended visual embeddings requires the text encoder and source tokens.")

        cond_stage = clip.cond_stage_model
        if hasattr(cond_stage, "clip") and isinstance(cond_stage.clip, str) and hasattr(cond_stage, cond_stage.clip):
            clip_model = getattr(cond_stage, cond_stage.clip)
        elif hasattr(cond_stage, "clip_model"):
            clip_model = cond_stage.clip_model
        elif hasattr(cond_stage, "clip_d"):
            clip_model = cond_stage.clip_d
        else:
            clip_model = cond_stage

        raw_visuals = []
        for key in active_keys:
            if key not in tokens_dict:
                raise ValueError(f"Missing source tokens for raw visual embedding {key}.")
            token_list = tokens_dict[key][next(iter(tokens_dict[key]))]
            tokens_only = [[token[0] for token in batch] for batch in token_list]
            embeds, _, _, _ = clip_model.process_tokens(tokens_only, device)
            v_start, v_end = visual_ranges[key]
            raw_visuals.append(embeds[0, v_start:v_end, :].clone().cpu())

        blended_raw = fuse_visual_token_sources(raw_visuals, visual_fusion_config, device, mask_cache, expected_visual_length)
        save_blended_visual_embeddings([blended_raw], visual_fusion_config, embedding_key)

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
    if not active_keys:
        raise ValueError("At least one sequence tensor is required.")
    tensors_list = [sequence_tensors[k] for k in active_keys]
    if any(t.ndim != 3 for t in tensors_list):
        raise ValueError("Every sequence tensor must have [batch, tokens, channels] shape.")
    if len({(t.shape[0], t.shape[2]) for t in tensors_list}) != 1:
        raise ValueError("All sequence tensors must have matching batch and channel dimensions.")

    B = tensors_list[0].shape[0]
    D = tensors_list[0].shape[2]

    blend_preset = blend_config.get("blend_preset", "baseline")
    if blend_preset == "off":
        first_pooled = pooled_tensors.get(active_keys[0]) if pooled_tensors else None
        return tensors_list[0], first_pooled
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

                stacked_norm = torch.nn.functional.normalize(stacked, p=2, dim=1, eps=1e-8)
                consensus_norm = torch.nn.functional.normalize(consensus, p=2, dim=0, eps=1e-8)
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
                        safe_sims = stretched_sims[mask].clamp(min=0.0, max=1.0)
                        row_weights[mask] = torch.pow(safe_sims, power_alpha) * torch.pow((distance_base - safe_sims).clamp(min=0.0), diversity_beta)
                    else:
                        row_weights[mask] = torch.pow(stretched_sims[mask].clamp(min=0.0), power_alpha)
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

                stacked_norm = torch.nn.functional.normalize(stacked, p=2, dim=1, eps=1e-8)
                consensus_norm = torch.nn.functional.normalize(consensus, p=2, dim=0, eps=1e-8)
                similarities = torch.mv(stacked_norm, consensus_norm)

                row_weights = torch.zeros_like(similarities)
                mask = similarities >= similarity_threshold

                if mask.any():
                    if diversity_beta > 0.0:
                        safe_sims = similarities[mask].clamp(min=0.0, max=1.0)
                        row_weights[mask] = torch.pow(safe_sims, power_alpha) * torch.pow((1.001 - safe_sims).clamp(min=0.0), diversity_beta)
                    else:
                        row_weights[mask] = torch.pow(similarities[mask].clamp(min=0.0), power_alpha)
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

            if blend_method == "linear":
                merged_p = torch.mean(stacked_p, dim=0) * global_scale
                P_blended_batches.append(merged_p)
                continue

            stacked_p_norm = torch.nn.functional.normalize(stacked_p, p=2, dim=1, eps=1e-8)
            consensus_p_norm = torch.nn.functional.normalize(consensus_p, p=2, dim=0, eps=1e-8)
            similarities_p = torch.mv(stacked_p_norm, consensus_p_norm)

            weights_p = torch.zeros_like(similarities_p)
            mask_p = similarities_p >= similarity_threshold

            if mask_p.any():
                if diversity_beta > 0.0:
                    safe_sims = similarities_p[mask_p].clamp(min=0.0, max=1.0)
                    weights_p[mask_p] = torch.pow(safe_sims, power_alpha) * torch.pow((1.001 - safe_sims).clamp(min=0.0), diversity_beta)
                else:
                    weights_p[mask_p] = torch.pow(similarities_p[mask_p].clamp(min=0.0), power_alpha)
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
    key_name = next(iter(tokens.keys()))
    token_list = tokens[key_name][0]
    if not any(is_image_token(token) for token in token_list):
        return 0, 0
    mapping = build_token_to_conditioning_map(token_list, cond_tensor)
    for i, t in enumerate(token_list):
        if is_image_token(t):
            return mapping[i][0], mapping[i][1]

    return 0, 0

def encode_embedding_classical_scaled_bias(clip, text, llama_template=None, **kwargs):
    if clip is None:
        raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

    if "(" not in text or ")" not in text:
        tokens = clip.tokenize(text, llama_template=llama_template, **kwargs)
        return clip.encode_from_tokens_scheduled(tokens)

    clean_text = ""
    biases_to_apply = []
    for segment, strength in token_weights(text, 1.0):
        start_tokens = clip.tokenize(clean_text, llama_template=llama_template, **kwargs)
        key_name = next(iter(start_tokens.keys()))
        start_count = len(start_tokens[key_name][0])
        clean_text += segment
        end_tokens = clip.tokenize(clean_text, llama_template=llama_template, **kwargs)
        end_count = len(end_tokens[key_name][0])
        if strength != 1.0 and end_count > start_count:
            if not math.isfinite(strength):
                raise ValueError("Contextual prompt weights must be finite.")
            biases_to_apply.append({"start": start_count, "end": end_count, "strength": float(strength)})

    tokens = clip.tokenize(clean_text, llama_template=llama_template, **kwargs)
    conditioning = clip.encode_from_tokens_scheduled(tokens)

    if not biases_to_apply:
        return conditioning

    # Apply contextual vector scaling directly to each schedule. This is a
    # custom operation for modern encoders that disable Core prompt weights.
    new_conditioning = []

    for i in range(len(conditioning)):
        cond, cond_dict = conditioning[i]
        new_cond = cond.clone()

        key_name = next(iter(tokens.keys()))
        token_list = tokens[key_name][0]
        mapping = build_token_to_conditioning_map(token_list, new_cond)

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

            if start < 0 or start >= end:
                continue

            new_cond[:, start:end, :] *= strength

        new_conditioning.append([new_cond, cond_dict.copy()])

    return new_conditioning


def strip_contextual_weight_syntax(text: str) -> str:
    """Return the exact clean text consumed by contextual vector scaling."""
    if "(" not in text or ")" not in text:
        return text
    return "".join(segment for segment, _ in token_weights(text, 1.0))

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

def extract_and_flatten_images(image_inputs) -> tuple:
    """
    Extracts individual images from batched image tensors within the image_inputs dict.
    Returns:
        - raw_images: Dict mapping sequential index to a single image tensor of shape [1, H, W, C]
        - flat_images: List of individual image tensors
        - is_zero_indexed: Boolean indicating if the original keys started at 0
    """
    is_zero_indexed = False
    if image_inputs is not None:
        for k in image_inputs.keys():
            digits = re.findall(r'\d+', k)
            if digits and int(digits[0]) == 0:
                is_zero_indexed = True
                break

    flat_images = []
    if image_inputs is not None:
        # Sort keys numerically by their suffix to ensure correct sequential order
        def get_num(k):
            digits = re.findall(r'\d+', k)
            return int(digits[0]) if digits else 0
        sorted_keys = sorted(image_inputs.keys(), key=get_num)

        for k in sorted_keys:
            v = image_inputs[k]
            if v is not None:
                if isinstance(v, torch.Tensor) and len(v.shape) == 4:
                    # Shape is [B, H, W, C]. Slice into B individual tensors of [1, H, W, C]
                    for i in range(v.shape[0]):
                        flat_images.append(v[i:i+1])
                elif isinstance(v, list):
                    # Handle lists of tensors if passed
                    for item in v:
                        if isinstance(item, torch.Tensor) and len(item.shape) == 4:
                            for i in range(item.shape[0]):
                                flat_images.append(item[i:i+1])
                        elif isinstance(item, torch.Tensor):
                            flat_images.append(item)
                else:
                    flat_images.append(v)

    start_idx = 0 if is_zero_indexed else 1
    raw_images = {}
    for idx, img in enumerate(flat_images):
        raw_images[start_idx + idx] = img

    return raw_images, flat_images, is_zero_indexed

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
    q, k = self.qknorm(q, k)
    if freqs is not None:
        q, k = apply_rope(q, k, freqs)
    if self.kvheads != self.heads:
        rep = self.heads // self.kvheads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    bias = None
    if weights and any(kb != 0.0 for _, kb in weights):
        bias = q.new_zeros(1, 1, 1, k.shape[2])
        for pos, kb in weights:
            if kb != 0.0 and pos < bias.shape[-1]:
                bias[..., pos] = kb
    if bias is not None:
        if mask is None:
            mask = bias
        else:
            if mask.dtype == torch.bool:
                additive_mask = torch.zeros(mask.shape, device=mask.device, dtype=q.dtype)
                additive_mask.masked_fill_(~mask, -torch.finfo(q.dtype).max)
            else:
                additive_mask = mask.to(dtype=q.dtype)
            mask = additive_mask + bias.to(device=mask.device, dtype=q.dtype)
    out = optimized_attention(q, k, v, self.heads, mask=mask, skip_reshape=True, transformer_options=transformer_options)
    return self.wo(out * torch.sigmoid(gate))

