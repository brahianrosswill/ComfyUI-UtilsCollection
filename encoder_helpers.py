import ast
from contextlib import contextmanager
import hashlib
import operator
import os
import math
import re
import torch
import logging
import numbers
import threading
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
from comfy.utils import common_upscale

from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention
from comfy.sd1_clip import token_weights


_VISUAL_ENCODER_PATH_LOCK = threading.RLock()


def prepare_vae_reference_image(samples, target_size, dimension_multiple, upscale_method="bicubic"):
    """Resize BCHW image samples for VAE encoding with configurable alignment."""
    multiple = int(dimension_multiple)
    if multiple < 4:
        raise ValueError("VAE dimension multiple must be at least 4.")
    height, width = samples.shape[-2:]
    if target_size is None:
        target_width = width
        target_height = height
    else:
        total_pixels = int(target_size) ** 2
        scale = math.sqrt(total_pixels / (width * height))
        target_width = width * scale
        target_height = height * scale
    aligned_width = max(multiple, round(target_width / multiple) * multiple)
    aligned_height = max(multiple, round(target_height / multiple) * multiple)
    return common_upscale(samples, aligned_width, aligned_height, upscale_method, "disabled")


def _resolve_clip_transformer(clip):
    stage = getattr(clip, "cond_stage_model", None)
    if stage is None:
        return None
    if hasattr(stage, "clip") and isinstance(stage.clip, str) and hasattr(stage, stage.clip):
        clip_model = getattr(stage, stage.clip)
    elif hasattr(stage, "clip_model"):
        clip_model = stage.clip_model
    elif hasattr(stage, "clip_d"):
        clip_model = stage.clip_d
    else:
        clip_model = stage
    return getattr(clip_model, "transformer", None)


@contextmanager
def qwen3vl_visual_encoder_path(clip, path: str):
    """Select current grid/DeepStack or pre-d0008a89 flat Qwen3-VL encoding."""
    if path == "grid-deepstack":
        with _VISUAL_ENCODER_PATH_LOCK:
            yield
        return
    if path != "legacy-flat":
        raise ValueError(f"Unsupported visual encoder path: {path}")

    transformer = _resolve_clip_transformer(clip)
    if transformer is None or not hasattr(transformer, "build_image_inputs"):
        raise ValueError("legacy-flat requires a Qwen3-VL text encoder with build_image_inputs support.")

    # Core d0008a89 made Qwen3-VL build grid MRoPE, a visual-position mask,
    # and DeepStack inputs here. Returning empty inputs reproduces the inherited
    # pre-update BaseLlama forward while leaving image preprocessing unchanged.
    with _VISUAL_ENCODER_PATH_LOCK:
        original = transformer.build_image_inputs
        transformer.build_image_inputs = lambda embeds, embeds_info: (None, None, None)
        try:
            logging.warning("Visual fusion is using the pre-d0008a89 legacy flat Qwen3-VL encoder path.")
            yield
        finally:
            transformer.build_image_inputs = original


def _encode_scheduled_with_visual_path(clip, tokens, visual_encoder_path: str):
    with qwen3vl_visual_encoder_path(clip, visual_encoder_path):
        return clip.encode_from_tokens_scheduled(tokens)


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


def _released_krea2_prefix_end(token_list, expanded_length: int) -> int:
    """Mirror released Core's Krea2 prefix slice exactly."""
    template_end = -1
    count_im_start = 0
    ids = [_token_id(token) for token in token_list]
    for index, token_id in enumerate(ids):
        if token_id == _QWEN_IM_START and count_im_start < 2:
            template_end = index
            count_im_start += 1

    if template_end < 0:
        raise ValueError("Could not locate the Krea2 template prefix marker used by released Core.")
    if expanded_length > template_end + 3:
        if ids[template_end + 1:template_end + 3] == [_QWEN_USER, _QWEN_NL]:
            template_end += 3
    return template_end


def _qwen3vl_resized_dimensions(height: int, width: int) -> tuple[int, int]:
    """Replicate released Core's Qwen3-VL resize arithmetic locally."""
    patch_size = 16
    merge_size = 2
    min_pixels = 3136
    max_pixels = 12845056
    factor = patch_size * merge_size
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor

    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / beta / factor) * factor)
        resized_width = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor

    return resized_height, resized_width


def _qwen3vl_image_span(token) -> int | None:
    value = token[0] if isinstance(token, tuple) and token else token
    if not isinstance(value, dict) or value.get("type") != "image":
        return None
    image = value.get("data")
    if not torch.is_tensor(image) or image.ndim != 4:
        return None
    height, width = image.shape[1:3]
    resized_height, resized_width = _qwen3vl_resized_dimensions(height, width)
    return (resized_height // 16) * (resized_width // 16) // 4


def qwen3vl_visual_grid(image) -> tuple[int, int]:
    """Return the exact post-patch, post-merge Qwen visual token grid."""
    if not torch.is_tensor(image) or image.ndim != 4:
        raise ValueError("Visual token layout error: processed image must have shape [batch, height, width, channels].")
    resized_height, resized_width = _qwen3vl_resized_dimensions(*image.shape[1:3])
    return resized_height // 32, resized_width // 32


def visual_fusion_grid(image, visual_length: int, legacy_flat: bool = False) -> tuple[int, int]:
    """Describe the usable visual layout without inventing legacy spatial coordinates."""
    if visual_length < 1:
        raise ValueError("Visual token layout error: visual range must contain at least one token.")
    if legacy_flat:
        return 1, visual_length
    grid = qwen3vl_visual_grid(image)
    if grid[0] * grid[1] != visual_length:
        raise ValueError(f"Visual token layout error: grid {grid} does not match range length {visual_length}.")
    return grid


def build_token_to_conditioning_map(token_list, cond_tensor) -> list[tuple[int, int]]:
    """Map raw tokenizer entries to conditioning spans, validating all inferred lengths."""
    cond_len = cond_tensor.shape[1]
    is_krea2 = cond_tensor.shape[-1] == 12 * 2560
    exact_spans = [_qwen3vl_image_span(token) if is_image_token(token) else 1 for token in token_list]
    if not all(span is not None for span in exact_spans):
        raise ValueError("Cannot derive token positions because an image token has no usable Qwen3-VL tensor payload.")

    total_length = sum(exact_spans)
    prefix_len = _released_krea2_prefix_end(token_list, total_length) if is_krea2 else 0
    if any(is_image_token(token) for token in token_list[:prefix_len]):
        raise ValueError("Released Core's Krea2 prefix slice crosses a visual token; mapping is unsafe.")

    token_spans = exact_spans[prefix_len:]
    expected_length = sum(token_spans)
    if expected_length != cond_len:
        image_details = [
            (index, exact_spans[index])
            for index, token in enumerate(token_list)
            if is_image_token(token)
        ]
        nearby_ids = [_token_id(token) for token in token_list[max(0, prefix_len - 3):prefix_len + 4]]
        raise ValueError(
            "Released Core's Krea2 prefix rule does not match the returned conditioning length; "
            "refusing to guess a visual range "
            f"(conditioning_length={cond_len}, expected_length={expected_length}, "
            f"expanded_length={total_length}, prefix_end={prefix_len}, "
            f"image_spans={image_details}, nearby_token_ids={nearby_ids})."
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


def _spatial_perturbation_seed(seed: int) -> int:
    payload = f"utils-collection-spatial-perturbation:{seed}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _perturb_spatial_assignments(mask: torch.Tensor, amount: float, seed: int) -> torch.Tensor:
    """Exchange differently labelled cells without changing any source count."""
    if amount <= 0.0 or mask.numel() < 2:
        return mask

    flat = mask.flatten().clone()
    requested_pairs = int(flat.numel() * amount) // 2
    if requested_pairs < 1:
        return mask

    generator = torch.Generator(device=flat.device).manual_seed(_spatial_perturbation_seed(seed))
    randomized = torch.randperm(flat.numel(), generator=generator, device=flat.device).tolist()
    labels = flat.tolist()
    buckets = {}
    for index in randomized:
        buckets.setdefault(labels[index], []).append(index)

    changed_pairs = 0
    while changed_pairs < requested_pairs:
        available = [label for label, indices in buckets.items() if indices]
        if len(available) < 2:
            break
        available.sort(key=lambda label: (-len(buckets[label]), label))
        first_label, second_label = available[:2]
        first_index = buckets[first_label].pop()
        second_index = buckets[second_label].pop()
        first_value = flat[first_index].clone()
        flat[first_index] = flat[second_index]
        flat[second_index] = first_value
        changed_pairs += 1
    return flat.reshape(mask.shape)


def _cleanup_primary_pairs(mask: torch.Tensor) -> torch.Tensor:
    """Swap complementary primary islands and holes while preserving source counts."""
    h, w = mask.shape
    primary = mask.eq(0)
    padded = F.pad(primary, (1, 1, 1, 1))
    neighbors = torch.stack([
        padded[row:row + h, column:column + w]
        for row in range(3)
        for column in range(3)
        if (row, column) != (1, 1)
    ])
    isolated = (primary & ~neighbors.any(dim=0)).flatten().nonzero().flatten()
    holes = (~primary & neighbors.all(dim=0)).flatten().nonzero().flatten()
    pair_count = min(isolated.numel(), holes.numel())
    if pair_count == 0:
        return mask

    flat = mask.flatten().clone()
    island_indices = isolated[:pair_count]
    hole_indices = holes[:pair_count]
    hole_values = flat[hole_indices].clone()
    flat[hole_indices] = flat[island_indices]
    flat[island_indices] = hole_values
    return flat.reshape(mask.shape)


def generate_spatial_fusion_mask(N: int, num_sources: int, method: str, block_size: int = 2, dither_ratio: float = 0.5, device: str = "cpu", seed: int = 0, grid_shape=None, dither_secondary_pattern: str = "checkerboard", dither_mask_cleanup: bool = False, spatial_perturbation: float = 0.0) -> torch.Tensor:
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
    if not 0.0 <= spatial_perturbation <= 1.0:
        raise ValueError("Spatial perturbation must be between 0.0 and 1.0.")
    if not 0 <= seed <= 0xffffffffffffffff:
        raise ValueError("Visual fusion seed must be between 0 and 18446744073709551615.")
    if num_sources == 1:
        return torch.zeros(N, dtype=torch.long, device=device)

    h, w = grid_shape if grid_shape is not None else reconstruct_2d_grid(N)
    if h < 1 or w < 1 or h * w != N:
        raise ValueError(f"Visual token layout error: grid {grid_shape} does not contain {N} tokens.")
    rows = torch.arange(h, device=device).unsqueeze(1)
    columns = torch.arange(w, device=device).unsqueeze(0)

    if method == "spatial-checkerboard":
        mask = (rows + columns) % num_sources
    elif method == "spatial-block-interleave":
        mask = (rows // block_size + columns // block_size) % num_sources
    else:
        if dither_secondary_pattern not in {"checkerboard", "block-interleave"}:
            raise ValueError(f"Unsupported dither secondary pattern: {dither_secondary_pattern}")
        if block_size < 1:
            raise ValueError("Visual block size must be at least 1.")
        generator = torch.Generator(device=device).manual_seed(seed)
        random = torch.rand(N, generator=generator, device=device)
        if dither_secondary_pattern == "block-interleave":
            secondary = rows // block_size + columns // block_size
        else:
            secondary = rows + columns
        other_sources = 1 + (secondary % (num_sources - 1)).flatten()
        mask = torch.where(random < dither_ratio, 0, other_sources).reshape(h, w)

    mask = _perturb_spatial_assignments(mask, spatial_perturbation, seed)
    if method == "spatial-dither-random" and dither_mask_cleanup and 0.0 < dither_ratio < 1.0:
        mask = _cleanup_primary_pairs(mask)
    return mask.flatten()


def _visual_fusion_mask(config, grid_shape, num_sources, mask_device, output_device, mask_cache):
    method = config.get("visual_fusion_method", "spatial-checkerboard")
    block_size = config.get("visual_block_size", 2)
    dither_ratio = config.get("dither_ratio", 0.5)
    seed = config.get("seed", 0)
    secondary = config.get("dither_secondary_pattern", "checkerboard")
    cleanup = config.get("dither_mask_cleanup", False)
    perturbation = config.get("spatial_perturbation", 0.0)
    key = (tuple(grid_shape), num_sources, method, secondary, cleanup, perturbation, block_size, dither_ratio, seed)
    if key not in mask_cache:
        mask_cache[key] = generate_spatial_fusion_mask(grid_shape[0] * grid_shape[1], num_sources, method, block_size, dither_ratio, mask_device, seed, grid_shape, secondary, cleanup, perturbation)
    return mask_cache[key].to(output_device)


def fuse_visual_token_sources(sources, visual_fusion_config, mask_device, mask_cache=None, expected_length=None, source_grids=None):
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
    if source_grids is None or len(source_grids) != len(sources):
        raise ValueError("Visual token layout error: every fusion source requires an explicit grid.")
    grids = [tuple(grid) for grid in source_grids]
    for source, grid in zip(sources, grids):
        if len(grid) != 2 or grid[0] < 1 or grid[1] < 1 or grid[0] * grid[1] != source.shape[0]:
            raise ValueError(f"Visual token layout error: grid {grid} is inconsistent with {source.shape[0]} tokens.")
    canonical_grid = grids[0]
    canonical_length = canonical_grid[0] * canonical_grid[1]
    if expected_length is not None and canonical_length != expected_length:
        raise ValueError(f"Visual token layout mismatch: expected {expected_length} tokens, received canonical grid {canonical_grid}.")

    output_dtype = sources[0].dtype
    compute_float = method == "linear"
    aligned = []
    for source, grid in zip(sources, grids):
        value = source.to(dtype=torch.float32) if compute_float else source
        if grid != canonical_grid:
            value = F.interpolate(value.reshape(grid[0], grid[1], -1).permute(2, 0, 1)[None], size=canonical_grid, mode="nearest")[0].permute(1, 2, 0).reshape(canonical_length, -1)
        aligned.append(value)

    stacked = torch.stack(aligned, dim=1)
    if method == "linear":
        return stacked.mean(dim=1).to(dtype=output_dtype)

    if mask_cache is None:
        mask_cache = {}
    mask = _visual_fusion_mask(visual_fusion_config, canonical_grid, len(sources), mask_device, stacked.device, mask_cache)
    fused = torch.take_along_dim(stacked, mask[:, None, None], dim=1).squeeze(1)
    return fused.to(dtype=output_dtype)


def fuse_deepstack_layers(deepstack_tensors, visual_fusion_config, device, mask_cache, expected_length, source_grids):
    active_keys = sorted(deepstack_tensors)
    if not active_keys:
        return None

    num_layers = len(deepstack_tensors[active_keys[0]])
    if any(len(deepstack_tensors[key]) != num_layers for key in active_keys[1:]):
        raise ValueError("Visual fusion sources produced different DeepStack layer counts.")

    blended = []
    for layer in range(num_layers):
        sources = [deepstack_tensors[key][layer].to(device=device) for key in active_keys]
        blended.append(fuse_visual_token_sources(sources, visual_fusion_config, device, mask_cache, expected_length, source_grids))
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
    visual_grids: dict = None,
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
    if visual_grids is None:
        raise ValueError("Visual token layout error: visual grids are required for fusion.")
    if mask_cache is None:
        mask_cache = {}

    active_keys = sorted(list(sequence_tensors.keys()))
    tensors_list = [sequence_tensors[k] for k in active_keys]
    if not tensors_list:
        return None, None

    B = tensors_list[0].shape[0]
    if any(key not in visual_grids for key in active_keys):
        raise ValueError("Visual token layout error: every fusion source requires a grid.")
    source_grids = [visual_grids[key] for key in active_keys]
    expected_visual_length = source_grids[0][0] * source_grids[0][1]
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
        blended_vis_2d = fuse_visual_token_sources(sources, visual_fusion_config, device, mask_cache, expected_visual_length, source_grids)

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

        blended_raw = fuse_visual_token_sources(raw_visuals, visual_fusion_config, device, mask_cache, expected_visual_length, source_grids)
        save_blended_visual_embeddings([blended_raw], visual_fusion_config, embedding_key)

    # Pooled output is kept pure from reference pass since text is identical
    ref_key = active_keys[0]
    P_blended = pooled_tensors.get(ref_key, None)

    return C_blended, P_blended


POWER_BLEND_PRESET = {
    "method": "consensus",
    "type": "median",
    "align": "similarity",
    "alignment_threshold": 0.9,
    "thresh": 0.75,
    "alpha": 8.0,
    "beta": 0.0,
    "norm": True,
    "scale": 1.0,
    "dsc": True,
    "soft_comfort": False,
}


def blend_text_vectors(sequence_tensors: dict, blend_config: dict, pooled_tensors: dict = None, device=None, compute_dtype=None) -> tuple:
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
    if device is None:
        device = tensors_list[0].device
    if compute_dtype is None:
        compute_dtype = tensors_list[0].dtype

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
        "dsc_high_diversity_concept": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.0, "thresh": 0.0, "beta": 3.0, "scale": 0.7, "norm": True, "dsc": True, "soft_comfort": True},
        "power_blend": POWER_BLEND_PRESET,
    }

    dsc_enabled = False
    soft_comfort_enabled = False

    if blend_preset != "off" and blend_preset in presets:
        p = presets[blend_preset]
        blend_method = p["method"]
        consensus_type = p["type"]
        alignment_method = p["align"]
        alignment_threshold = p.get("alignment_threshold", alignment_threshold)
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
        batch_tensors = [comfy.model_management.cast_to_device(t[b], device, compute_dtype) for t in tensors_list]

        if blend_method == "linear":
            max_len = max(t.shape[0] for t in batch_tensors)
            padded = []
            for t in batch_tensors:
                if t.shape[0] < max_len:
                    padding = t.new_zeros((max_len - t.shape[0], D))
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

            merged_seq = ref_tensor.new_zeros((N_ref, D))
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
            merged_seq = batch_tensors[0].new_zeros((max_len, D))
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

    C_blended = comfy.model_management.cast_to_device(
        torch.stack(C_blended_list, dim=0), tensors_list[0].device, tensors_list[0].dtype
    )

    # Blend metadata pooled outputs
    P_blended = None
    if pooled_tensors and any(p is not None for p in pooled_tensors.values()):
        pooled_list_active = [pooled_tensors[k] for k in active_keys if pooled_tensors.get(k) is not None]
        pooled_reference = pooled_list_active[0]
        P_blended_batches = []
        for b in range(B):
            stacked_p = torch.stack([
                comfy.model_management.cast_to_device(pooled[b], device, compute_dtype)
                for pooled in pooled_list_active
            ])
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
        P_blended = comfy.model_management.cast_to_device(
            torch.stack(P_blended_batches, dim=0), pooled_reference.device, pooled_reference.dtype
        )

    return C_blended, P_blended


def find_visual_token_range(tokens, cond_tensor, legacy_krea_spatial=False) -> tuple:
    key_name = next(iter(tokens.keys()))
    token_list = tokens[key_name][0]
    if not any(is_image_token(token) for token in token_list):
        return 0, 0

    if legacy_krea_spatial and cond_tensor.shape[-1] == 12 * 2560:
        image_indices = [index for index, token in enumerate(token_list) if is_image_token(token)]
        if len(image_indices) != 1:
            raise ValueError("Legacy Krea2 spatial mapping requires exactly one image per encoder pass.")
        text_count = len(token_list) - 1
        visual_length = cond_tensor.shape[1] - text_count
        if visual_length <= 0:
            raise ValueError("Legacy Krea2 spatial mapping produced a non-positive visual span.")
        visual_start = image_indices[0]
        visual_end = visual_start + visual_length
        trailing_text = len(token_list) - image_indices[0] - 1
        if visual_end + trailing_text != cond_tensor.shape[1]:
            raise ValueError("Legacy Krea2 spatial mapping does not cover the conditioning sequence.")
        return visual_start, visual_end

    mapping = build_token_to_conditioning_map(token_list, cond_tensor)
    for i, t in enumerate(token_list):
        if is_image_token(t):
            return mapping[i][0], mapping[i][1]

    return 0, 0

def encode_embedding_classical_scaled_bias(clip, text, llama_template=None, visual_encoder_path="grid-deepstack", **kwargs):
    if clip is None:
        raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

    if "(" not in text or ")" not in text:
        tokens = clip.tokenize(text, llama_template=llama_template, **kwargs)
        return _encode_scheduled_with_visual_path(clip, tokens, visual_encoder_path)

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
    conditioning = _encode_scheduled_with_visual_path(clip, tokens, visual_encoder_path)

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

