import json
import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from comfy_api.latest import io, ui
from nodes import MAX_RESOLUTION
from .helper_functions import resize_nchw
from .model_assets import require_huggingface_model
from .staged_face_helpers import (
    detect_many_or_warn,
    load_face_model,
)
from .staged_compositor_helpers import (
    RetainedStageCache,
    cached_layer_preview,
    projective_warp,
)


FaceDetectionType = io.Custom("FACE_DETECTION_MODEL")
FaceCompositeOptionsType = io.Custom("UC_FACE_COMPOSITE_OPTIONS")
StagedBackgroundOptionsType = io.Custom("UC_STAGED_LAYERED_BACKGROUND_OPTIONS")
StagedFaceOptionsType = io.Custom("UC_STAGED_MEDIAPIPE_FACE_OPTIONS")

_RESIZE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
_COMPOSITE_RESIZE_METHODS = ["auto", *_RESIZE_METHODS]
_DEFAULT_LAYER_PLACEMENT = {
    "scale": 0.9,
    "long_axis_shift": 0.0,
    "short_axis_shift": 0.0,
}
_DEFAULT_LAYER_PLACEMENT_V2 = {"scale": 0.9, "center_x": 0.5, "center_y": 0.5}
_MASK_THRESHOLD_TOOLTIP = "Minimum background-removal confidence retained as foreground before cleanup."
_BORDER_CLEANUP_TOOLTIP = "Source-edge strip width in pixels where weak foreground predictions are removed; 0 disables it."
_ARTIFACT_CLEANUP_TOOLTIP = "Opening radius in pixels used to remove small or thin mask artifacts; 0 disables it."
_GAP_FILL_TOOLTIP = "Closing radius in pixels used to fill small mask cracks and holes; 0 disables it."
_FEATHER_TOOLTIP = "Inward mask-edge softness in pixels; 0 keeps the resized edge unchanged."
_IMAGE_RESIZE_TOOLTIP = (
    "Foreground resampling method. auto uses FP32 area reduction when shrinking and bicubic when enlarging; "
    "choose another method to override it."
)
_MASK_RESIZE_TOOLTIP = (
    "Mask resampling method. auto uses area when shrinking and bilinear when enlarging while preserving soft coverage; "
    "nearest-exact produces a hard binary edge."
)
_BACKGROUND_REMOVAL_MODEL_FILES = {
    "birefnet": "birefnet.safetensors",
    "lucida": "lucida.safetensors",
}
_LUCIDA_IMAGE_MEAN = [0.485, 0.456, 0.406]
_LUCIDA_IMAGE_STD = [0.229, 0.224, 0.225]
_MISSING = object()
_INTERNAL_BACKGROUND_REMOVAL_CACHE = {"key": None, "model": None}


def _load_internal_background_removal_model(model_name):
    selected = str(model_name or "birefnet").lower()
    filename = _BACKGROUND_REMOVAL_MODEL_FILES.get(selected)
    if filename is None:
        choices = ", ".join(_BACKGROUND_REMOVAL_MODEL_FILES)
        raise ValueError(f"Unsupported internal background-removal model {model_name!r}; choose {choices}.")

    from comfy import bg_removal_model

    model_path = require_huggingface_model(
        "background_removal",
        filename,
        "Comfy-Org/BiRefNet",
        f"background_removal/{filename}",
    )

    cache_key = (selected, os.path.normcase(os.path.abspath(model_path)))
    if _INTERNAL_BACKGROUND_REMOVAL_CACHE["key"] == cache_key:
        return _INTERNAL_BACKGROUND_REMOVAL_CACHE["model"]

    try:
        model = bg_removal_model.load(model_path)
    except Exception as exc:
        raise ValueError(
            f"Comfy Core could not load {filename} as the {selected} background-removal model."
        ) from exc
    if model is None or not callable(getattr(model, "encode_image", None)):
        raise ValueError(
            f"Comfy Core did not recognize {filename} as a supported background-removal model."
        )

    if selected == "lucida":
        model.image_size = 1024
        model.image_mean = list(_LUCIDA_IMAGE_MEAN)
        model.image_std = list(_LUCIDA_IMAGE_STD)
        if isinstance(getattr(model, "config", None), dict):
            model.config.update({
                "image_size": model.image_size,
                "image_mean": list(model.image_mean),
                "image_std": list(model.image_std),
            })

    _INTERNAL_BACKGROUND_REMOVAL_CACHE["key"] = cache_key
    _INTERNAL_BACKGROUND_REMOVAL_CACHE["model"] = model
    return model


def _resize_image(image, width, height, method, crop="disabled"):
    return resize_nchw(image.movedim(-1, 1), width, height, method, crop).movedim(1, -1)


def _resize_mask(mask, width, height, method, crop="disabled"):
    return resize_nchw(mask.unsqueeze(1), width, height, method, crop).squeeze(1)


def _composite_resize_method(method, source_width, source_height, width, height, mask=False):
    if method not in _COMPOSITE_RESIZE_METHODS:
        raise ValueError(f"Unsupported composite resize method: {method!r}.")
    if method != "auto":
        return method
    shrinking = width < source_width or height < source_height
    reducing_both_axes = width <= source_width and height <= source_height
    if mask:
        return "area" if reducing_both_axes and shrinking else "bilinear"
    return "area" if reducing_both_axes and shrinking else "bicubic"


def _resize_composite_image(image, width, height, method):
    source_height, source_width = image.shape[1:3]
    if (source_width, source_height) == (width, height):
        return image
    selected = _composite_resize_method(method, source_width, source_height, width, height)
    return _resize_image(image, width, height, selected)


def _resize_composite_mask(mask, width, height, method):
    source_height, source_width = mask.shape[-2:]
    if (source_width, source_height) == (width, height):
        resized = mask
        selected = method
    else:
        selected = _composite_resize_method(method, source_width, source_height, width, height, mask=True)
        resized = _resize_mask(mask, width, height, selected)
    if selected == "nearest-exact":
        return (resized >= 0.5).to(resized)
    return resized.clamp(0.0, 1.0)


def _broadcast_batch(value, batch_size, name):
    if value.shape[0] == batch_size:
        return value
    if value.shape[0] == 1:
        return value.expand(batch_size, *value.shape[1:])
    raise ValueError(f"{name} batch size must be 1 or {batch_size}.")


def _blur_mask(mask, radius):
    if radius <= 0:
        return mask
    sigma = max(float(radius) / 3.0, 0.1)
    kernel_radius = max(1, int(math.ceil(sigma * 3.0)))
    coordinates = torch.arange(-kernel_radius, kernel_radius + 1, device=mask.device, dtype=mask.dtype)
    kernel = torch.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    mask = F.conv2d(mask.unsqueeze(1), kernel.view(1, 1, 1, -1), padding=(0, kernel_radius))
    mask = F.conv2d(mask, kernel.view(1, 1, -1, 1), padding=(kernel_radius, 0))
    return mask.squeeze(1)


def _expand_mask(mask, amount):
    if amount == 0:
        return mask
    radius = abs(int(amount))
    kernel = 2 * radius + 1
    mask = mask.unsqueeze(0).unsqueeze(0)
    if amount > 0:
        mask = F.max_pool2d(mask, kernel, stride=1, padding=radius)
    else:
        mask = 1.0 - F.max_pool2d(1.0 - mask, kernel, stride=1, padding=radius)
    return mask[0, 0]


def _feather_mask(mask, radius):
    if radius == 0:
        return mask
    radius = int(radius)
    if radius < 0:
        contracted = _expand_mask(mask, -max(1, math.ceil(abs(radius) / 2)))
        blurred = _blur_mask(contracted.unsqueeze(0), abs(radius))[0].clamp(0.0, 1.0)
        return torch.minimum(mask, blurred)
    blurred = _blur_mask(mask.unsqueeze(0), radius)[0].clamp(0.0, 1.0)
    return torch.maximum(mask, blurred)


def _binary_dilate(mask, radius):
    radius = max(0, int(radius))
    if radius == 0:
        return mask
    kernel = radius * 2 + 1
    padded = F.pad(mask[None, None], (radius, radius, radius, radius), value=0.0)
    return F.max_pool2d(padded, kernel, stride=1)[0, 0]


def _binary_erode(mask, radius):
    radius = max(0, int(radius))
    if radius == 0:
        return mask
    kernel = radius * 2 + 1
    padded = F.pad(mask[None, None], (radius, radius, radius, radius), value=0.0)
    return 1.0 - F.max_pool2d(1.0 - padded, kernel, stride=1)[0, 0]


def _refine_foreground_mask(raw_mask, threshold, border_cleanup_width, artifact_cleanup_radius, gap_fill_radius):
    raw_mask = raw_mask.clamp(0.0, 1.0)
    mask = (raw_mask >= threshold).to(raw_mask)
    border_width = min(max(0, int(border_cleanup_width)), min(mask.shape) // 2)
    if border_width:
        height, width = mask.shape
        rows = torch.arange(height, device=mask.device)
        columns = torch.arange(width, device=mask.device)
        border = (
            (rows[:, None] < border_width)
            | (rows[:, None] >= height - border_width)
            | (columns[None, :] < border_width)
            | (columns[None, :] >= width - border_width)
        )
        strong_threshold = min(1.0, float(threshold) + 0.25)
        mask = mask * (~(border & (raw_mask < strong_threshold))).to(mask)
    if artifact_cleanup_radius:
        mask = _binary_dilate(_binary_erode(mask, artifact_cleanup_radius), artifact_cleanup_radius)
    if gap_fill_radius:
        mask = _binary_erode(_binary_dilate(mask, gap_fill_radius), gap_fill_radius)
    return (mask >= 0.5).to(raw_mask)


def _flatten_autogrow_images(image_inputs):
    images = []
    for key in sorted(image_inputs or {}, key=lambda value: int("".join(filter(str.isdigit, value)) or 0)):
        value = image_inputs[key]
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item is None:
                continue
            if not torch.is_tensor(item) or item.ndim != 4:
                raise ValueError(f"Foreground input {key} must have shape [batch, height, width, channels].")
            images.extend(item[index:index + 1] for index in range(item.shape[0]))
    return images


def _foreground_input_order(key):
    digits = "".join(filter(str.isdigit, key))
    return int(digits or 0), key


def _ordered_single_foregrounds(image_inputs):
    foregrounds = []
    for key in sorted(image_inputs or {}, key=_foreground_input_order):
        value = image_inputs[key]
        values = value if isinstance(value, (list, tuple)) else [value]
        values = [item for item in values if item is not None]
        if len(values) != 1 or not torch.is_tensor(values[0]) or values[0].ndim != 4:
            raise ValueError(f"Foreground input {key} must contain exactly one image tensor.")
        image = values[0]
        if image.shape[0] != 1:
            raise ValueError(f"Foreground input {key} must contain exactly one image, not a batch.")
        foregrounds.append((key, image))
    return foregrounds


def _parse_layer_payload(value):
    if value is None or value == "":
        value = {}
    if isinstance(value, str):
        if value.strip() in _COMPOSITE_RESIZE_METHODS:
            logging.warning(
                "Detected legacy layered-composite widget ordering; using default placement data. "
                "Reload the workflow in the frontend once to migrate its saved values."
            )
            value = {}
        else:
            try:
                value = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError(f"Layer placement data is not valid JSON: {error.msg}.") from error
    if not isinstance(value, dict):
        raise ValueError("Layer placement data must be a JSON object.")
    version = value.get("version", 1)
    if version not in (1, 2, 3):
        raise ValueError(f"Unsupported layer placement data version: {version}.")
    layers = value.get("layers", {})
    if not isinstance(layers, dict):
        raise ValueError("Layer placement data 'layers' must be a JSON object.")
    layer_order = value.get("layer_order", [])
    if not isinstance(layer_order, list) or any(not isinstance(key, str) for key in layer_order):
        raise ValueError("Layer placement data 'layer_order' must be an array of socket names.")
    workspace_padding = value.get("workspace_padding", 0.5)
    if isinstance(workspace_padding, bool):
        raise ValueError("Layer placement workspace_padding must be numeric.")
    try:
        workspace_padding = float(workspace_padding)
    except (TypeError, ValueError) as error:
        raise ValueError("Layer placement workspace_padding must be numeric.") from error
    if not math.isfinite(workspace_padding) or not 0.0 <= workspace_padding <= 1.0:
        raise ValueError("Layer placement workspace_padding must be between 0 and 1.")
    return version, layers, layer_order, workspace_padding


def _parse_layer_placements(value):
    version, layers, _, _ = _parse_layer_payload(value)

    parsed = {}
    for key, placement in layers.items():
        if not isinstance(key, str) or not isinstance(placement, dict):
            raise ValueError("Every layer placement must be an object keyed by its foreground socket name.")
        result = dict(_DEFAULT_LAYER_PLACEMENT_V2 if version >= 2 else _DEFAULT_LAYER_PLACEMENT)
        fields = [("scale", 0.05, 10.0)]
        if version >= 2:
            fields.extend((("center_x", -10.0, 10.0), ("center_y", -10.0, 10.0)))
        else:
            fields.extend((("long_axis_shift", -1.0, 1.0), ("short_axis_shift", -1.0, 1.0)))
        for field, minimum, maximum in fields:
            raw = placement.get(field, result[field])
            if isinstance(raw, bool):
                raise ValueError(f"Layer {key} field {field} must be numeric.")
            try:
                number = float(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Layer {key} field {field} must be numeric.") from error
            if not math.isfinite(number) or number < minimum or number > maximum:
                raise ValueError(f"Layer {key} field {field} must be between {minimum} and {maximum}.")
            result[field] = number
        for flip_field in ("flip_horizontal", "flip_vertical"):
            flip_value = placement.get(flip_field, False)
            if not isinstance(flip_value, bool):
                raise ValueError(f"Layer {key} field {flip_field} must be Boolean.")
            result[flip_field] = flip_value
        if version == 3:
            rotation = placement.get("rotation", 0.0)
            if isinstance(rotation, bool):
                raise ValueError(f"Layer {key} field rotation must be numeric.")
            try:
                rotation = float(rotation)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Layer {key} field rotation must be numeric.") from error
            if not math.isfinite(rotation):
                raise ValueError(f"Layer {key} field rotation must be finite.")
            result["rotation"] = ((rotation + 180.0) % 360.0) - 180.0
            corners = placement.get("corners", [[-1, -1], [1, -1], [1, 1], [-1, 1]])
            result["corners"] = _validate_quad(corners, key)
            included = placement.get("included", True)
            if not isinstance(included, bool):
                raise ValueError(f"Layer {key} field included must be Boolean.")
            result["included"] = included
        result["_version"] = version
        parsed[key] = result
    return parsed


def _validate_quad(corners, key="face"):
    if not isinstance(corners, list) or len(corners) != 4:
        raise ValueError(f"Layer {key} corners must contain four points.")
    points = []
    for point in corners:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"Layer {key} corners must contain [x, y] points.")
        x, y = point
        if isinstance(x, bool) or isinstance(y, bool):
            raise ValueError(f"Layer {key} corner coordinates must be numeric.")
        x, y = float(x), float(y)
        if not math.isfinite(x) or not math.isfinite(y) or not (-1 <= x <= 1 and -1 <= y <= 1):
            raise ValueError(f"Layer {key} corner coordinates must be within [-1, 1].")
        points.append([x, y])
    cross = []
    for index in range(4):
        a, b, c = points[index], points[(index + 1) % 4], points[(index + 2) % 4]
        cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    area = abs(sum(
        points[i][0] * points[(i + 1) % 4][1] - points[(i + 1) % 4][0] * points[i][1]
        for i in range(4)
    )) * 0.5
    if area < 1e-4 or not (all(value > 1e-6 for value in cross) or all(value < -1e-6 for value in cross)):
        raise ValueError(f"Layer {key} corners must form a convex, non-zero-area quadrilateral.")
    return points


def _ordered_layer_keys(value, available_keys):
    _, _, requested, _ = _parse_layer_payload(value)
    available = list(available_keys)
    available_set = set(available)
    ordered = []
    for key in requested:
        if key in available_set and key not in ordered:
            ordered.append(key)
    ordered.extend(key for key in available if key not in ordered)
    return ordered


def _placement_offsets(background_width, background_height, placed_width, placed_height, placement, workspace_padding=0.0):
    if placement.get("_version") == 2 or "center_x" in placement or "center_y" in placement:
        offset_x = round(float(placement.get("center_x", 0.5)) * background_width - placed_width / 2.0)
        offset_y = round(float(placement.get("center_y", 0.5)) * background_height - placed_height / 2.0)
        padding_x = background_width * 0.25 * workspace_padding
        padding_y = background_height * 0.25 * workspace_padding
        x_limits = (-padding_x, background_width + padding_x - placed_width, 0.0, background_width - placed_width)
        y_limits = (-padding_y, background_height + padding_y - placed_height, 0.0, background_height - placed_height)
        offset_x = round(min(max(offset_x, min(x_limits)), max(x_limits)))
        offset_y = round(min(max(offset_y, min(y_limits)), max(y_limits)))
        return offset_x, offset_y
    long_shift = (placement["long_axis_shift"] + 1.0) / 2.0
    short_shift = (placement["short_axis_shift"] + 1.0) / 2.0
    padding_x = background_width * 0.25 * workspace_padding
    padding_y = background_height * 0.25 * workspace_padding
    travel_x = background_width + 2.0 * padding_x - placed_width
    travel_y = background_height + 2.0 * padding_y - placed_height
    if background_width > background_height:
        offset_x = round(-padding_x + travel_x * long_shift)
        offset_y = round(-padding_y + travel_y * short_shift)
    elif background_height > background_width:
        offset_y = round(-padding_y + travel_y * long_shift)
        offset_x = round(-padding_x + travel_x * short_shift)
    else:
        offset_x = round(-padding_x + travel_x * long_shift)
        offset_y = round(-padding_y + travel_y * short_shift)
    return offset_x, offset_y


def _visible_placement_slices(background_width, background_height, placed_width, placed_height, offset_x, offset_y):
    destination_top = max(0, offset_y)
    destination_bottom = min(background_height, offset_y + placed_height)
    destination_left = max(0, offset_x)
    destination_right = min(background_width, offset_x + placed_width)
    if destination_bottom <= destination_top or destination_right <= destination_left:
        return None
    source_top = destination_top - offset_y
    source_bottom = source_top + destination_bottom - destination_top
    source_left = destination_left - offset_x
    source_right = source_left + destination_right - destination_left
    return (
        destination_top, destination_bottom, destination_left, destination_right,
        source_top, source_bottom, source_left, source_right,
    )


def _save_editor_preview(image, prefix):
    saved = ui.ImageSaveHelper.save_images(
        image,
        filename_prefix=prefix,
        folder_type=io.FolderType.temp,
        cls=None,
        compress_level=1,
    )
    return dict(saved[0]) if saved else None


def _crop_bounds(mask, padding, multiple=8):
    points = torch.nonzero(mask > 0, as_tuple=False)
    if points.numel() == 0:
        raise ValueError("Mask is empty.")
    height, width = mask.shape[-2:]
    min_y = int(points[:, -2].min())
    max_y = int(points[:, -2].max()) + 1
    min_x = int(points[:, -1].min())
    max_x = int(points[:, -1].max()) + 1
    side = max(max_x - min_x, max_y - min_y) + 2 * padding
    side = min(max(height, width), ((side + multiple - 1) // multiple) * multiple)
    crop_width = min(side, width)
    crop_height = min(side, height)
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    x = max(0, min(center_x - crop_width // 2, width - crop_width))
    y = max(0, min(center_y - crop_height // 2, height - crop_height))
    return x, y, crop_width, crop_height


class UC_CropByMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_CropByMask",
            display_name="Crop By Mask",
            category="utils/image",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Int.Input("padding", default=64, min=0, max=MAX_RESOLUTION, step=8),
                io.Int.Input(
                    "multiple",
                    default=8,
                    min=4,
                    max=256,
                    step=4,
                    tooltip="Expand the crop dimensions to this pixel multiple without resizing the image or mask.",
                ),
            ],
            outputs=[
                io.Image.Output("image"),
                io.Mask.Output("mask"),
                io.Int.Output("crop_x", display_name="X"),
                io.Int.Output("crop_y", display_name="Y"),
                io.Int.Output("crop_width", display_name="Width"),
                io.Int.Output("crop_height", display_name="Height"),
            ],
        )

    @classmethod
    def execute(cls, image, mask, padding, multiple=8):
        if mask.shape[-2:] != image.shape[1:3]:
            mask = _resize_mask(mask, image.shape[2], image.shape[1], "nearest-exact")
        mask = _broadcast_batch(mask, image.shape[0], "Mask")
        x, y, width, height = _crop_bounds(mask, int(padding), int(multiple))
        return io.NodeOutput(image[:, y:y + height, x:x + width], mask[:, y:y + height, x:x + width], x, y, width, height)


class UC_ImageCropMerge(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ImageCropMerge",
            display_name="Image Crop Merge",
            category="utils/image",
            inputs=[
                io.Image.Input("cropped_image"),
                io.Image.Input("original_image"),
                io.Int.Input("crop_x", default=0, min=0, max=MAX_RESOLUTION, force_input=True),
                io.Int.Input("crop_y", default=0, min=0, max=MAX_RESOLUTION, force_input=True),
                io.Int.Input("crop_width", default=512, min=1, max=MAX_RESOLUTION, force_input=True),
                io.Int.Input("crop_height", default=512, min=1, max=MAX_RESOLUTION, force_input=True),
                io.Combo.Input("resize_method", options=_RESIZE_METHODS, default="lanczos"),
                io.Mask.Input("mask", optional=True),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, cropped_image, original_image, crop_x, crop_y, crop_width, crop_height, resize_method, mask=None):
        result = original_image.clone()
        source = _resize_image(cropped_image, int(crop_width), int(crop_height), resize_method).to(result)
        source = _broadcast_batch(source, result.shape[0], "Cropped image")
        x1 = max(0, int(crop_x))
        y1 = max(0, int(crop_y))
        x2 = min(result.shape[2], int(crop_x) + int(crop_width))
        y2 = min(result.shape[1], int(crop_y) + int(crop_height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Crop coordinates do not overlap the original image.")
        source = source[:, y1 - int(crop_y):y2 - int(crop_y), x1 - int(crop_x):x2 - int(crop_x)]
        if mask is None:
            result[:, y1:y2, x1:x2] = source
        else:
            mask = _resize_mask(mask, int(crop_width), int(crop_height), "bilinear").to(result)
            mask = _broadcast_batch(mask, result.shape[0], "Mask")
            mask = mask[:, y1 - int(crop_y):y2 - int(crop_y), x1 - int(crop_x):x2 - int(crop_x)].clamp(0.0, 1.0).unsqueeze(-1)
            result[:, y1:y2, x1:x2] = result[:, y1:y2, x1:x2] * (1.0 - mask) + source * mask
        return io.NodeOutput(result)


class UC_ImageAndMaskResize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ImageAndMaskResize",
            display_name="Image and Mask Resize",
            category="utils/image",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Image.Input("target"),
                io.Combo.Input("resize_method", options=_RESIZE_METHODS, default="lanczos"),
                io.Combo.Input("crop", options=["disabled", "center"], default="disabled"),
                io.Int.Input("mask_blur_radius", default=0, min=0, max=256, step=1),
                io.Int.Input("width", default=512, min=1, max=MAX_RESOLUTION, force_input=True, optional=True),
                io.Int.Input("height", default=512, min=1, max=MAX_RESOLUTION, force_input=True, optional=True),
            ],
            outputs=[io.Image.Output(), io.Mask.Output()],
        )

    @classmethod
    def execute(cls, image, mask, target, resize_method, crop, mask_blur_radius, width=None, height=None):
        target_width = int(width) if width is not None else target.shape[2]
        target_height = int(height) if height is not None else target.shape[1]
        if mask.shape[-2:] != image.shape[1:3]:
            mask = _resize_mask(mask, image.shape[2], image.shape[1], "bilinear")
        image = _resize_image(image, target_width, target_height, resize_method, crop)
        mask = _resize_mask(mask, target_width, target_height, "bilinear", crop)
        mask = _blur_mask(mask, int(mask_blur_radius)).clamp(0.0, 1.0)
        return io.NodeOutput(image, mask)


class UC_ResizeMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ResizeMask",
            display_name="Resize Mask",
            category="utils/mask",
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input("width", default=512, min=0, max=MAX_RESOLUTION, step=1),
                io.Int.Input("height", default=512, min=0, max=MAX_RESOLUTION, step=1),
                io.Boolean.Input("keep_proportions", default=False),
                io.Combo.Input("upscale_method", options=_RESIZE_METHODS, default="bilinear"),
                io.Combo.Input("crop", options=["disabled", "center"], default="disabled"),
            ],
            outputs=[io.Mask.Output(), io.Int.Output("width"), io.Int.Output("height")],
        )

    @classmethod
    def execute(cls, mask, width, height, keep_proportions, upscale_method, crop):
        original_height, original_width = mask.shape[-2:]
        width = original_width if width == 0 else int(width)
        height = original_height if height == 0 else int(height)
        if keep_proportions:
            ratio = min(width / original_width, height / original_height)
            width = max(1, round(original_width * ratio))
            height = max(1, round(original_height * ratio))
        mask = _resize_mask(mask, width, height, upscale_method, crop)
        return io.NodeOutput(mask, mask.shape[2], mask.shape[1])


def _ordered_ring(edges):
    adjacency = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    start = next(iter(adjacency))
    ring = [start]
    previous = None
    current = start
    while True:
        next_index = next((index for index in adjacency[current] if index != previous), None)
        if next_index is None or next_index == start:
            break
        ring.append(next_index)
        previous, current = current, next_index
    return ring


def _polygon_mask(height, width, points, device, dtype):
    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).polygon([(float(x), float(y)) for x, y in points], fill=255)
    return torch.from_numpy(np.asarray(image).copy()).to(device=device, dtype=dtype).div_(255.0)


def _expanded_box(box, padding, width, height):
    x1 = max(0, math.floor(float(box[0])) - padding)
    y1 = max(0, math.floor(float(box[1])) - padding)
    x2 = min(width, math.ceil(float(box[2])) + padding)
    y2 = min(height, math.ceil(float(box[3])) + padding)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Detected face has an empty bounding box.")
    return x1, y1, x2, y2


def _largest_face(faces, name):
    if not faces:
        raise ValueError(f"No face was detected in the {name} image.")
    return max(faces, key=lambda face: max(0.0, float(face["bbox_xyxy"][2] - face["bbox_xyxy"][0])) * max(0.0, float(face["bbox_xyxy"][3] - face["bbox_xyxy"][1])))


def _similarity_transform(source_points, target_points):
    source_points = np.asarray(source_points, dtype=np.float32)
    target_points = np.asarray(target_points, dtype=np.float32)
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    centered_source = source_points - source_center
    centered_target = target_points - target_center
    covariance = centered_source.T @ centered_target
    left, singular_values, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        singular_values[-1] *= -1
        rotation = right.T @ left.T
    scale = float(singular_values.sum() / max(float((centered_source * centered_source).sum()), 1e-6))
    translation = target_center - scale * (source_center @ rotation.T)
    return scale, rotation.astype(np.float32), translation.astype(np.float32)


def _transform_source(source, oval, foreground, output_height, output_width, scale, rotation, translation):
    device = source.device
    yy, xx = torch.meshgrid(
        torch.arange(output_height, device=device, dtype=torch.float32),
        torch.arange(output_width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    output_points = torch.stack((xx, yy), dim=-1)
    rotation = torch.as_tensor(rotation, device=device, dtype=torch.float32)
    translation = torch.as_tensor(translation, device=device, dtype=torch.float32)
    source_points = ((output_points - translation) @ rotation) / max(float(scale), 1e-6)
    grid_x = (source_points[..., 0] + 0.5) * (2.0 / source.shape[1]) - 1.0
    grid_y = (source_points[..., 1] + 0.5) * (2.0 / source.shape[0]) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    layers = torch.cat((source.movedim(-1, 0), oval.unsqueeze(0), foreground.unsqueeze(0)), dim=0).unsqueeze(0)
    transformed = F.grid_sample(layers, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    return transformed[:3].movedim(0, -1), transformed[3], transformed[4]


def _smoothstep(value):
    value = value.clamp(0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _add_control(controls, values, seen, point, value):
    key = (round(float(point[0]), 2), round(float(point[1]), 2))
    if key not in seen:
        seen.add(key)
        controls.append(point)
        values.append(value)


def _warp_target(target, source_oval_points, target_oval_points, strength, decay_radius):
    if strength <= 0:
        return target
    device = target.device
    height, width = target.shape[:2]
    source_points = np.asarray(source_oval_points, dtype=np.float32)
    target_points = np.asarray(target_oval_points, dtype=np.float32)
    center = source_points.mean(axis=0)
    controls = []
    values = []
    seen = set()
    for source_point, target_point in zip(source_points, target_points):
        _add_control(controls, values, seen, source_point, (target_point - source_point) * float(strength))
    for source_point in source_points:
        direction = source_point - center
        length = np.linalg.norm(direction)
        if length > 0:
            fixed = source_point + direction * (float(decay_radius) / length)
            fixed[0] = np.clip(fixed[0], 0, width - 1)
            fixed[1] = np.clip(fixed[1], 0, height - 1)
            _add_control(controls, values, seen, fixed, np.zeros(2, dtype=np.float32))
    border_step = max(8, min(32, int(decay_radius) // 2))
    for x in range(0, width, border_step):
        _add_control(controls, values, seen, np.array([x, 0], dtype=np.float32), np.zeros(2, dtype=np.float32))
        _add_control(controls, values, seen, np.array([x, height - 1], dtype=np.float32), np.zeros(2, dtype=np.float32))
    for y in range(0, height, border_step):
        _add_control(controls, values, seen, np.array([0, y], dtype=np.float32), np.zeros(2, dtype=np.float32))
        _add_control(controls, values, seen, np.array([width - 1, y], dtype=np.float32), np.zeros(2, dtype=np.float32))
    for point in ((width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        _add_control(controls, values, seen, np.array(point, dtype=np.float32), np.zeros(2, dtype=np.float32))

    controls = torch.as_tensor(np.asarray(controls), device=device, dtype=torch.float32)
    values = torch.as_tensor(np.asarray(values), device=device, dtype=torch.float32)
    controls[:, 0] = (controls[:, 0] + 0.5) * (2.0 / width) - 1.0
    controls[:, 1] = (controls[:, 1] + 0.5) * (2.0 / height) - 1.0
    values[:, 0] *= 2.0 / width
    values[:, 1] *= 2.0 / height
    difference = controls[:, None] - controls[None]
    distance_squared = (difference * difference).sum(dim=-1)
    kernel = distance_squared * torch.log(distance_squared + 1e-6)
    kernel.diagonal().add_(1e-4)
    affine = torch.cat((torch.ones((controls.shape[0], 1), device=device), controls), dim=1)
    system = torch.cat((
        torch.cat((kernel, affine), dim=1),
        torch.cat((affine.T, torch.zeros((3, 3), device=device)), dim=1),
    ), dim=0)
    coefficients = torch.linalg.solve(system, torch.cat((values, torch.zeros((3, 2), device=device)), dim=0))

    grid_rows = []
    x_coordinates = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (2.0 / width) - 1.0
    for start in range(0, height, 64):
        end = min(start + 64, height)
        y_coordinates = (torch.arange(start, end, device=device, dtype=torch.float32) + 0.5) * (2.0 / height) - 1.0
        yy, xx = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
        points = torch.stack((xx, yy), dim=-1)
        difference = points.unsqueeze(-2) - controls
        distance_squared = (difference * difference).sum(dim=-1)
        basis = distance_squared * torch.log(distance_squared + 1e-6)
        point_affine = torch.cat((torch.ones((*points.shape[:-1], 1), device=device), points), dim=-1)
        displacement = basis @ coefficients[:-3] + point_affine @ coefficients[-3:]
        pixel_x = torch.arange(width, device=device, dtype=torch.float32)
        pixel_y = torch.arange(start, end, device=device, dtype=torch.float32)
        edge_x = torch.minimum(pixel_x, width - 1 - pixel_x).unsqueeze(0)
        edge_y = torch.minimum(pixel_y, height - 1 - pixel_y).unsqueeze(1)
        displacement *= _smoothstep(torch.minimum(edge_x, edge_y) / 2.0).unsqueeze(-1)
        grid_rows.append(points + displacement)
    grid = torch.cat(grid_rows, dim=0).unsqueeze(0)
    warped = F.grid_sample(target.movedim(-1, 0).unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False)
    return warped.squeeze(0).movedim(0, -1)


class UC_UnifiedBackgroundReplace(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        foreground_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("foreground"), prefix="foreground_", min=1, max=50
        )
        return io.Schema(
            node_id="UC_UnifiedBackgroundReplace",
            display_name="Unified Background Replace",
            category="utils/image",
            inputs=[
                io.BackgroundRemoval.Input("background_removal_model", tooltip="Core background-removal model used to isolate every foreground."),
                io.Image.Input("background", tooltip="Single image used as the shared output canvas."),
                io.Float.Input("foreground_scale", default=0.90, min=0.05, max=10.0, step=0.01, tooltip="Fraction of the background's shortest side occupied by the foreground's longest bound. Values above 1 overscale and crop at the canvas edges."),
                io.Float.Input("long_axis_shift", default=0.0, min=-1.0, max=1.0, step=0.01, tooltip="Position along the background's longest axis: -1 is left/up, 0 is centered, and 1 is right/down."),
                io.Float.Input("short_axis_shift", default=0.0, min=-1.0, max=1.0, step=0.01, tooltip="Position along the background's shortest axis: -1 is up/left, 0 is centered, and 1 is down/right."),
                io.Float.Input("mask_threshold", default=0.50, min=0.0, max=1.0, step=0.01, tooltip="Minimum model confidence retained as solid foreground."),
                io.Int.Input("border_cleanup_width", default=2, min=0, max=64, step=1, advanced=True, tooltip="Width of the source-edge strip where weak foreground predictions are removed."),
                io.Int.Input("artifact_cleanup_radius", default=2, min=0, max=64, step=1, advanced=True, tooltip="Opening radius used to remove small and thin mask artifacts."),
                io.Int.Input("gap_fill_radius", default=2, min=0, max=64, step=1, advanced=True, tooltip="Closing radius used to fill small cracks and holes in the foreground."),
                io.Int.Input("feather_radius", default=2, min=0, max=64, step=1, advanced=True, tooltip="Inward edge softness; the foreground interior remains fully opaque."),
                io.Combo.Input("image_resize_method", options=_COMPOSITE_RESIZE_METHODS, default="auto", advanced=True, tooltip=_IMAGE_RESIZE_TOOLTIP),
                io.Combo.Input("mask_resize_method", options=_COMPOSITE_RESIZE_METHODS, default="auto", advanced=True, tooltip=_MASK_RESIZE_TOOLTIP),
                io.Float.Input("workspace_padding", default=0.5, min=0.0, max=1.0, step=0.05, advanced=True, tooltip="Permitted off-canvas placement margin, up to 25% of each background axis."),
                io.Autogrow.Input("foreground_images", template=foreground_template, tooltip="Images to isolate, resize, center, and composite in socket order."),
            ],
            outputs=[
                io.Image.Output("images"),
                io.Mask.Output("masks"),
            ],
        )

    @classmethod
    def execute(
        cls,
        background_removal_model,
        background,
        foreground_images,
        foreground_scale,
        long_axis_shift,
        short_axis_shift,
        mask_threshold,
        border_cleanup_width,
        artifact_cleanup_radius,
        gap_fill_radius,
        feather_radius,
        image_resize_method="auto",
        mask_resize_method="auto",
        workspace_padding=0.5,
    ):
        if not torch.is_tensor(background) or background.ndim != 4 or background.shape[0] != 1:
            raise ValueError("Unified Background Replace requires exactly one background image.")
        if background.shape[-1] < 3:
            raise ValueError("Background image must have at least three channels.")
        workspace_padding = float(workspace_padding)
        if not math.isfinite(workspace_padding) or not 0.0 <= workspace_padding <= 1.0:
            raise ValueError("Unified Background Replace workspace_padding must be between 0 and 1.")
        foregrounds = _flatten_autogrow_images(foreground_images)
        if not foregrounds:
            raise ValueError("Unified Background Replace requires at least one foreground image.")

        background = background[..., :3]
        background_height, background_width = background.shape[1:3]
        target_longest = max(1, round(min(background_height, background_width) * float(foreground_scale)))
        composites = []
        masks = []

        for index, foreground in enumerate(foregrounds, start=1):
            if foreground.shape[-1] < 3:
                raise ValueError(f"Foreground image {index} must have at least three channels.")
            foreground = foreground[..., :3]
            raw_mask = background_removal_model.encode_image(foreground)
            if not torch.is_tensor(raw_mask):
                raise ValueError(f"Background removal model returned an invalid mask for foreground image {index}.")
            if raw_mask.ndim == 4 and raw_mask.shape[1] == 1:
                raw_mask = raw_mask[:, 0]
            elif raw_mask.ndim == 4 and raw_mask.shape[-1] == 1:
                raw_mask = raw_mask[..., 0]
            if raw_mask.ndim != 3 or raw_mask.shape[0] != 1:
                raise ValueError(f"Background removal model must return one [batch, height, width] mask for foreground image {index}.")
            if raw_mask.shape[-2:] != foreground.shape[1:3]:
                raw_mask = _resize_composite_mask(raw_mask, foreground.shape[2], foreground.shape[1], mask_resize_method)
            refined = _refine_foreground_mask(
                raw_mask[0],
                float(mask_threshold),
                border_cleanup_width,
                artifact_cleanup_radius,
                gap_fill_radius,
            )
            points = torch.nonzero(refined > 0, as_tuple=False)
            if points.numel() == 0:
                raise ValueError(f"Background removal produced an empty foreground mask for image {index}.")
            top = int(points[:, 0].min())
            bottom = int(points[:, 0].max()) + 1
            left = int(points[:, 1].min())
            right = int(points[:, 1].max()) + 1
            crop = foreground[:, top:bottom, left:right]
            crop_mask = refined[None, top:bottom, left:right]
            crop_height, crop_width = crop.shape[1:3]
            scale = target_longest / max(crop_height, crop_width)
            placed_height = max(1, round(crop_height * scale))
            placed_width = max(1, round(crop_width * scale))
            resized_foreground = _resize_composite_image(crop, placed_width, placed_height, image_resize_method).to(background)
            resized_mask = _resize_composite_mask(crop_mask, placed_width, placed_height, mask_resize_method).to(background)
            resized_mask = resized_mask[0]
            alpha = _feather_mask(resized_mask, -int(feather_radius)) if feather_radius else resized_mask

            placement = {
                "long_axis_shift": float(long_axis_shift),
                "short_axis_shift": float(short_axis_shift),
            }
            offset_x, offset_y = _placement_offsets(
                background_width, background_height, placed_width, placed_height,
                placement, workspace_padding,
            )
            slices = _visible_placement_slices(
                background_width, background_height, placed_width, placed_height, offset_x, offset_y
            )
            composite = background.clone()
            canvas_mask = background.new_zeros((1, background_height, background_width))
            if slices is None:
                composites.append(composite)
                masks.append(canvas_mask)
                continue
            (
                destination_top, destination_bottom, destination_left, destination_right,
                source_top, source_bottom, source_left, source_right,
            ) = slices
            placed_alpha = alpha[source_top:source_bottom, source_left:source_right]
            placed_foreground = resized_foreground[0, source_top:source_bottom, source_left:source_right]
            region = composite[0, destination_top:destination_bottom, destination_left:destination_right]
            composite[0, destination_top:destination_bottom, destination_left:destination_right] = (
                region * (1.0 - placed_alpha.unsqueeze(-1)) + placed_foreground * placed_alpha.unsqueeze(-1)
            )
            canvas_mask[0, destination_top:destination_bottom, destination_left:destination_right] = placed_alpha
            composites.append(composite)
            masks.append(canvas_mask)

        return io.NodeOutput(torch.cat(composites, dim=0), torch.cat(masks, dim=0))


def _stage_layered_foregrounds(
    background_removal_model,
    foreground_images,
    mask_threshold,
    border_cleanup_width,
    artifact_cleanup_radius,
    gap_fill_radius,
    mask_resize_method="auto",
    placement_data=None,
    retain_full_alpha=False,
):
    foregrounds = _ordered_single_foregrounds(foreground_images)
    if not foregrounds:
        raise ValueError("Layered foreground staging requires at least one foreground image.")
    layers = []
    full_alpha_by_socket = {}
    placements = _parse_layer_placements(placement_data)
    for key, foreground in foregrounds:
        if foreground.shape[-1] < 3:
            raise ValueError(f"Foreground input {key} must have at least three channels.")
        embedded_alpha = foreground[..., 3] if foreground.shape[-1] >= 4 else None
        foreground = foreground[..., :3]
        uses_embedded_alpha = embedded_alpha is not None
        if embedded_alpha is not None:
            # A supplied alpha channel is an explicit foreground matte. Preserve
            # it exactly instead of replacing it with a removal-model estimate.
            refined = embedded_alpha[0].to(device=foreground.device, dtype=foreground.dtype).clamp(0.0, 1.0)
            full_alpha = refined
        else:
            raw_mask = background_removal_model.encode_image(foreground)
            if not torch.is_tensor(raw_mask):
                raise ValueError(f"Background removal returned an invalid mask for {key}.")
            if raw_mask.ndim == 4 and raw_mask.shape[1] == 1:
                raw_mask = raw_mask[:, 0]
            elif raw_mask.ndim == 4 and raw_mask.shape[-1] == 1:
                raw_mask = raw_mask[..., 0]
            if raw_mask.ndim != 3 or raw_mask.shape[0] != 1:
                raise ValueError(f"Background removal must return one [batch, height, width] mask for {key}.")
            if raw_mask.shape[-2:] != foreground.shape[1:3]:
                raw_mask = _resize_composite_mask(raw_mask, foreground.shape[2], foreground.shape[1], mask_resize_method)
            full_alpha = raw_mask[0].to(device=foreground.device, dtype=foreground.dtype).clamp(0.0, 1.0)
            refined = _refine_foreground_mask(
                raw_mask[0],
                float(mask_threshold),
                border_cleanup_width,
                artifact_cleanup_radius,
                gap_fill_radius,
            )
        if retain_full_alpha:
            full_alpha_by_socket[key] = full_alpha
        points = torch.nonzero(refined > 0, as_tuple=False)
        if points.numel() == 0:
            raise ValueError(f"Background removal produced an empty foreground mask for {key}.")
        top = int(points[:, 0].min())
        bottom = int(points[:, 0].max()) + 1
        left = int(points[:, 1].min())
        right = int(points[:, 1].max()) + 1
        flip_horizontal = placements.get(key, {}).get("flip_horizontal", False)
        flip_vertical = placements.get(key, {}).get("flip_vertical", False)
        cropped_image = foreground[:, top:bottom, left:right]
        cropped_mask = refined[None, top:bottom, left:right]
        if flip_horizontal:
            cropped_image = torch.flip(cropped_image, dims=(2,))
            cropped_mask = torch.flip(cropped_mask, dims=(2,))
        if flip_vertical:
            cropped_image = torch.flip(cropped_image, dims=(1,))
            cropped_mask = torch.flip(cropped_mask, dims=(1,))
        layers.append({
            "socket": key,
            "image": cropped_image,
            "mask": cropped_mask,
            "flip_horizontal": flip_horizontal,
            "flip_vertical": flip_vertical,
            "uses_embedded_alpha": uses_embedded_alpha,
        })
    staged = {"version": 1, "layers": layers}
    return (staged, full_alpha_by_socket) if retain_full_alpha else staged


def _stage_face_foregrounds(
    background_removal_model,
    face_detection_model,
    foreground_images,
    background_options,
    face_options,
    placement_data=None,
):
    staged, full_alpha_by_socket = _stage_layered_foregrounds(
        background_removal_model, foreground_images,
        background_options["mask_threshold"], background_options["border_cleanup_width"],
        background_options["artifact_cleanup_radius"], background_options["gap_fill_radius"],
        background_options["mask_resize_method"], placement_data, retain_full_alpha=True,
    )
    ordinary = {layer["socket"]: layer for layer in staged["layers"]}
    result, warning_count = [], 0
    ring = _ordered_ring(face_detection_model.connection_sets["face_oval"])
    foregrounds = _ordered_single_foregrounds(foreground_images)
    detection_inputs = []
    for key, original in foregrounds:
        rgb = original[..., :3]
        detection_inputs.append((
            key,
            rgb.mul(255).add(0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()[0],
        ))
    detected_by_socket, failed_sockets = detect_many_or_warn(
        face_detection_model,
        detection_inputs,
        face_options["maximum_faces"],
        face_options["detection_threshold"],
    )
    warning_count = len(failed_sockets)

    for key, original in foregrounds:
        result.append(ordinary[key])
        rgb = original[..., :3]
        if key in failed_sockets:
            continue
        full_alpha = full_alpha_by_socket[key]
        faces = detected_by_socket[key]
        for face_index, face in enumerate(faces):
            x1, y1, x2, y2 = _expanded_box(
                face["bbox_xyxy"], face_options["bbox_expansion"], rgb.shape[2], rgb.shape[1]
            )
            points = face["landmarks_xy"][ring] - np.asarray([x1, y1], dtype=np.float32)
            crop_mask = _polygon_mask(
                y2 - y1, x2 - x1, points, rgb.device, rgb.dtype
            )
            crop_mask = crop_mask * full_alpha[y1:y2, x1:x2].to(crop_mask)
            crop_mask = _expand_mask(crop_mask, face_options["mask_expansion"]).clamp(0, 1)[None]
            if not torch.any(crop_mask > 0):
                continue
            socket = f"{key}_face_{face_index}"
            result.append({
                "socket": socket,
                "image": rgb[:, y1:y2, x1:x2],
                "mask": crop_mask,
                "is_face": True,
                "parent_socket": key,
                "uses_embedded_alpha": True,
                "default_scale": face_options["initial_face_scale"],
                "feather_radius": face_options["face_feather_radius"],
            })
    staged["layers"] = result
    staged["face_warning_count"] = warning_count
    return staged


def _apply_staged_layer_options(staged, foreground_blend, face_blend, face_feather_radius):
    foreground_factor = 0.5 + 0.5 * float(foreground_blend)
    face_factor = 0.5 + 0.5 * float(face_blend)
    return {
        **staged,
        "layers": [
            {
                **layer,
                "blend_factor": face_factor if layer.get("is_face") else foreground_factor,
                **({"feather_radius": int(face_feather_radius)} if layer.get("is_face") else {}),
            }
            for layer in staged["layers"]
        ],
    }


def _preview_staged_foregrounds(background, staged_foregrounds, feather_radius):
    if not torch.is_tensor(background) or background.ndim != 4 or background.shape[0] != 1:
        raise ValueError("Staged Layered Background Composite requires exactly one background image.")
    if background.shape[-1] < 3:
        raise ValueError("Background image must have at least three channels.")
    if not isinstance(staged_foregrounds, dict) or staged_foregrounds.get("version") != 1:
        raise ValueError("Staged foreground data is missing or incompatible.")
    layers = staged_foregrounds.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("Staged foreground data contains no layers.")
    background_height, background_width = background.shape[1:3]
    editor_metadata = {
        "version": 1,
        "stage_mode": "fresh",
        "background": {"width": background_width, "height": background_height},
        "layers": [],
    }
    if staged_foregrounds.get("background_removal_model_name"):
        editor_metadata["background_removal_model_name"] = staged_foregrounds["background_removal_model_name"]
    for layer in layers:
        crop = layer["image"]
        layer_feather = int(layer.get("feather_radius", feather_radius))
        applies_feather = bool(
            layer_feather and (layer.get("is_face") or not layer.get("uses_embedded_alpha", False))
        )
        entry = {
            "socket": layer["socket"],
            "crop_width": crop.shape[2],
            "crop_height": crop.shape[1],
            "flip_horizontal": bool(layer.get("flip_horizontal", False)),
            "flip_vertical": bool(layer.get("flip_vertical", False)),
            "is_face": bool(layer.get("is_face", False)),
            "blend_factor": float(layer.get("blend_factor", 1.0)),
        }
        try:
            def build_preview_tensor():
                alpha = layer["mask"][0]
                if applies_feather:
                    alpha = _feather_mask(alpha, -layer_feather)
                return torch.cat((crop[0], alpha.unsqueeze(-1)), dim=-1).unsqueeze(0)

            entry["preview"] = cached_layer_preview(
                staged_foregrounds,
                layer["socket"],
                ("rgba-v1", layer_feather if applies_feather else 0),
                build_preview_tensor,
                lambda image: _save_editor_preview(image, f"UC_layered_{layer['socket']}"),
            )
        except Exception:
            logging.warning(
                "Unable to create staged editor cutout preview for %s.",
                layer["socket"],
                exc_info=True,
            )
        editor_metadata["layers"].append(entry)
    passthrough = background[..., :3]
    empty_mask = passthrough.new_zeros((1, background_height, background_width))
    return io.NodeOutput(
        passthrough,
        empty_mask,
        [],
        passthrough.new_zeros((0, background_height, background_width)),
        ui={"uc_layered_scene_editor": [editor_metadata]},
    )


def _composite_staged_foregrounds(
    background,
    staged_foregrounds,
    placement_data,
    feather_radius,
    stage_mode=None,
    image_resize_method="auto",
    mask_resize_method="auto",
):
    if not torch.is_tensor(background) or background.ndim != 4 or background.shape[0] != 1:
        raise ValueError("Staged Layered Background Composite requires exactly one background image.")
    if background.shape[-1] < 3:
        raise ValueError("Background image must have at least three channels.")
    if not isinstance(staged_foregrounds, dict) or staged_foregrounds.get("version") != 1:
        raise ValueError("Staged foreground data is missing or incompatible.")
    layers = staged_foregrounds.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("Staged foreground data contains no layers.")
    placements = _parse_layer_placements(placement_data)
    placement_version, _, _, workspace_padding = _parse_layer_payload(placement_data)
    layers_by_socket = {layer["socket"]: layer for layer in layers}
    layers = [
        layers_by_socket[key]
        for key in _ordered_layer_keys(placement_data, layers_by_socket)
    ]
    scene = background[..., :3].clone()
    background_height, background_width = scene.shape[1:3]
    combined_mask = scene.new_zeros((1, background_height, background_width))
    layer_masks = []
    layer_boxes = []
    editor_layers = []

    for layer in layers:
        key = layer["socket"]
        crop = layer["image"]
        crop_mask = layer["mask"]
        preview_crop = crop
        preview_mask = crop_mask
        crop_height, crop_width = crop.shape[1:3]
        placement = placements.get(
            key,
            {**(_DEFAULT_LAYER_PLACEMENT_V2 if placement_version >= 2 else _DEFAULT_LAYER_PLACEMENT), "_version": placement_version},
        )
        if key not in placements and layer.get("default_scale") is not None:
            placement["scale"] = float(layer["default_scale"])
        excluded = not placement.get("included", True)
        staged_flip = bool(layer.get("flip_horizontal", False))
        desired_flip = bool(placement.get("flip_horizontal", False))
        if staged_flip != desired_flip:
            crop = torch.flip(crop, dims=(2,))
            crop_mask = torch.flip(crop_mask, dims=(2,))
        staged_flip_vertical = bool(layer.get("flip_vertical", False))
        desired_flip_vertical = bool(placement.get("flip_vertical", False))
        if staged_flip_vertical != desired_flip_vertical:
            crop = torch.flip(crop, dims=(1,))
            crop_mask = torch.flip(crop_mask, dims=(1,))
        target_longest = max(1, round(min(background_height, background_width) * placement["scale"]))
        scale = target_longest / max(crop_height, crop_width)
        placed_height = max(1, round(crop_height * scale))
        placed_width = max(1, round(crop_width * scale))
        resized_foreground = _resize_composite_image(crop, placed_width, placed_height, image_resize_method).to(scene)
        resized_mask = _resize_composite_mask(crop_mask, placed_width, placed_height, mask_resize_method).to(scene)
        if placement_version == 3:
            resized_foreground, resized_mask = projective_warp(
                resized_foreground, resized_mask,
                placement.get("corners", [[-1, -1], [1, -1], [1, 1], [-1, 1]]),
                placement.get("rotation", 0.0),
            )
            placed_height, placed_width = resized_foreground.shape[1:3]
        resized_mask = resized_mask[0]
        layer_feather = int(layer.get("feather_radius", feather_radius))
        placed_feather = (
            max(1, round(layer_feather * scale))
            if layer.get("is_face") and layer_feather
            else layer_feather
        )
        alpha = (
            resized_mask
            if (layer.get("uses_embedded_alpha", False) and not layer.get("is_face")) or not placed_feather
            else _feather_mask(resized_mask, -placed_feather)
        )
        offset_x, offset_y = _placement_offsets(
            background_width, background_height, placed_width, placed_height, placement,
            workspace_padding if placement_version >= 2 else 0.0,
        )
        slices = None if excluded else _visible_placement_slices(
            background_width, background_height, placed_width, placed_height, offset_x, offset_y
        )
        layer_mask = scene.new_zeros((background_height, background_width))
        layer_box = {"x": 0, "y": 0, "width": 0, "height": 0}
        if slices is not None:
            (
                destination_top, destination_bottom, destination_left, destination_right,
                source_top, source_bottom, source_left, source_right,
            ) = slices
            base_alpha = alpha[source_top:source_bottom, source_left:source_right]
            layer_mask[
                destination_top:destination_bottom,
                destination_left:destination_right,
            ] = base_alpha
            layer_box = {
                "x": destination_left,
                "y": destination_top,
                "width": destination_right - destination_left,
                "height": destination_bottom - destination_top,
            }
            mask_region = combined_mask[0, destination_top:destination_bottom, destination_left:destination_right]
            blend_factor = float(layer.get("blend_factor", 1.0))
            placed_alpha = base_alpha * (1.0 - mask_region * (1.0 - blend_factor))
            placed_foreground = resized_foreground[0, source_top:source_bottom, source_left:source_right]
            region = scene[0, destination_top:destination_bottom, destination_left:destination_right]
            scene[0, destination_top:destination_bottom, destination_left:destination_right] = (
                region * (1.0 - placed_alpha.unsqueeze(-1)) + placed_foreground * placed_alpha.unsqueeze(-1)
            )
            combined_mask[0, destination_top:destination_bottom, destination_left:destination_right] = (
                mask_region + base_alpha * (1.0 - mask_region)
            )
        layer_boxes.append(layer_box)
        layer_masks.append(layer_mask)
        editor_layers.append({
            "socket": key,
            "crop_width": crop_width,
            "crop_height": crop_height,
            "preview_crop": preview_crop,
            "preview_mask": preview_mask,
            "preview_feather": layer_feather,
            "preview_applies_feather": bool(
                layer_feather and (layer.get("is_face") or not layer.get("uses_embedded_alpha", False))
            ),
            "flip_horizontal": staged_flip,
            "flip_vertical": staged_flip_vertical,
            "is_face": bool(layer.get("is_face", False)),
            "included": not excluded,
            "blend_factor": float(layer.get("blend_factor", 1.0)),
        })

    editor_metadata = {
        "version": 1,
        "background": {"width": background_width, "height": background_height},
        "layers": [],
    }
    if staged_foregrounds.get("background_removal_model_name"):
        editor_metadata["background_removal_model_name"] = staged_foregrounds["background_removal_model_name"]
    if stage_mode:
        editor_metadata["stage_mode"] = stage_mode
    for layer in editor_layers:
        entry = {
            key: layer[key]
            for key in (
                "socket", "crop_width", "crop_height", "flip_horizontal", "flip_vertical",
                "is_face", "included", "blend_factor",
            )
        }
        try:
            def build_preview_tensor():
                alpha = layer["preview_mask"][0]
                if layer["preview_applies_feather"]:
                    alpha = _feather_mask(alpha, -layer["preview_feather"])
                return torch.cat(
                    (layer["preview_crop"][0], alpha.unsqueeze(-1)), dim=-1
                ).unsqueeze(0)

            entry["preview"] = cached_layer_preview(
                staged_foregrounds,
                layer["socket"],
                ("rgba-v1", layer["preview_feather"] if layer["preview_applies_feather"] else 0),
                build_preview_tensor,
                lambda image: _save_editor_preview(
                    image, f"UC_layered_{layer['socket']}"
                ),
            )
        except Exception:
            logging.warning("Unable to create staged editor cutout preview for %s.", layer["socket"], exc_info=True)
        editor_metadata["layers"].append(entry)
    ordered_masks = torch.stack(layer_masks) if layer_masks else scene.new_zeros(
        (0, background_height, background_width)
    )
    return io.NodeOutput(
        scene,
        combined_mask,
        [layer_boxes] if layer_boxes else [],
        ordered_masks,
        ui={"uc_layered_scene_editor": [editor_metadata]},
    )


class UC_StagedLayeredBackgroundCompositeOptions(io.ComfyNode):
    DEFAULTS = {
        "mask_threshold": 0.5, "border_cleanup_width": 2, "artifact_cleanup_radius": 2,
        "gap_fill_radius": 2, "feather_radius": 2,
        "image_resize_method": "auto", "mask_resize_method": "auto",
        "foreground_blend": 1.0,
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_StagedLayeredBackgroundCompositeOptions",
            display_name="Staged Composite Options",
            category="utils/image",
            inputs=[
                io.Float.Input("mask_threshold", default=0.5, min=0, max=1, step=0.01),
                io.Int.Input("border_cleanup_width", default=2, min=0, max=64),
                io.Int.Input("artifact_cleanup_radius", default=2, min=0, max=64),
                io.Int.Input("gap_fill_radius", default=2, min=0, max=64),
                io.Int.Input("feather_radius", default=2, min=0, max=64),
                io.Combo.Input("image_resize_method", options=_COMPOSITE_RESIZE_METHODS, default="auto"),
                io.Combo.Input("mask_resize_method", options=_COMPOSITE_RESIZE_METHODS, default="auto"),
                io.Float.Input(
                    "foreground_blend", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "1.0 is fully foreground; 0.0 is a 50/50 normal blend where another foreground or face "
                        "is underneath. Background-only areas remain fully foreground."
                    ),
                ),
            ],
            outputs=[StagedBackgroundOptionsType.Output(display_name="Background Options")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(cls.DEFAULTS | kwargs)


class UC_StagedMediaPipeFaceOptions(io.ComfyNode):
    DEFAULTS = {
        "detection_threshold": 0.25, "maximum_faces": 16, "bbox_expansion": 64,
        "mask_expansion": 0, "face_feather_radius": 8, "initial_face_scale": 0.25,
        "face_blend": 1.0,
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_StagedMediaPipeFaceOptions",
            display_name="Staged MediaPipe Face Options",
            category="utils/image",
            inputs=[
                io.Float.Input("detection_threshold", default=0.25, min=0, max=1, step=0.01),
                io.Int.Input("maximum_faces", default=16, min=1, max=16),
                io.Int.Input("bbox_expansion", default=64, min=0, max=MAX_RESOLUTION),
                io.Int.Input("mask_expansion", default=0, min=-MAX_RESOLUTION, max=MAX_RESOLUTION),
                io.Int.Input("face_feather_radius", default=8, min=0, max=512),
                io.Float.Input("initial_face_scale", default=0.25, min=0.05, max=10, step=0.01),
                io.Float.Input(
                    "face_blend", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "1.0 is fully face; 0.0 is a 50/50 normal blend where another foreground or face is "
                        "underneath. Background-only areas remain fully face."
                    ),
                ),
            ],
            outputs=[StagedFaceOptionsType.Output(display_name="Face Options")],
        )

    @classmethod
    def execute(cls, **kwargs):
        result = cls.DEFAULTS | kwargs
        result["maximum_faces"] = min(16, int(result["maximum_faces"]))
        return io.NodeOutput(result)


class UC_StagedMediaPipeFaceBackgroundComposite(io.ComfyNode):
    _staged_by_node = RetainedStageCache(max_entries=8)

    @classmethod
    def define_schema(cls):
        foreground_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("foreground", lazy=True), prefix="foreground_", min=1, max=50
        )
        return io.Schema(
            node_id="UC_StagedMediaPipeFaceBackgroundComposite",
            display_name="Face Background Composite",
            category="utils/image",
            inputs=[
                io.Image.Input("background"),
                StagedBackgroundOptionsType.Input("background_options", display_name="Background Options", optional=True),
                StagedFaceOptionsType.Input("face_options", display_name="Face Options", optional=True),
                io.Combo.Input("execution_mode", options=["run_staging", "run_staged", "full_run"], default="full_run"),
                io.String.Input("placement_data", default='{"version":3,"workspace_padding":0.5,"layers":{}}', advanced=True),
                io.Combo.Input("background_removal_model_name", options=["birefnet", "lucida"], default="birefnet"),
                io.Autogrow.Input("foreground_images", template=foreground_template),
            ],
            outputs=[
                io.Image.Output("image"),
                io.Mask.Output("mask"),
                io.BoundingBox.Output("bounding_boxes", display_name="Boxes"),
                io.Mask.Output("layer_masks", display_name="Layer Masks"),
            ],
            hidden=[io.Hidden.unique_id],
            is_output_node=True,
        )

    @classmethod
    def check_lazy_status(cls, execution_mode, foreground_images=None, **kwargs):
        if execution_mode == "run_staged":
            return []
        required = []
        for value in (foreground_images or {}).values():
            if isinstance(value, tuple) and len(value) == 2 and value[0] is None and value[1]:
                required.append(value[1])
        return required

    @classmethod
    def execute(
        cls, background, foreground_images, execution_mode, placement_data,
        background_removal_model_name="birefnet", background_options=None, face_options=None,
    ):
        node_id = str(cls.hidden.unique_id or "")
        background_options = UC_StagedLayeredBackgroundCompositeOptions.DEFAULTS | (background_options or {})
        face_options = UC_StagedMediaPipeFaceOptions.DEFAULTS | (face_options or {})
        if execution_mode == "run_staged":
            if node_id not in cls._staged_by_node:
                raise ValueError("No retained face-aware stage is available. Run staging first.")
            staged, stage_mode = cls._staged_by_node[node_id], "retained"
        elif execution_mode in ("run_staging", "full_run"):
            removal_model = _load_internal_background_removal_model(background_removal_model_name)
            face_model = load_face_model()
            staged = _stage_face_foregrounds(
                removal_model, face_model, foreground_images,
                background_options, face_options, placement_data,
            )
            staged["background_removal_model_name"] = str(background_removal_model_name).lower()
            cls._staged_by_node[node_id] = staged
            staged = cls._staged_by_node[node_id]
            if execution_mode == "run_staging":
                preview_stage = _apply_staged_layer_options(
                    staged,
                    background_options["foreground_blend"],
                    face_options["face_blend"],
                    face_options["face_feather_radius"],
                )
                return _preview_staged_foregrounds(background, preview_stage, background_options["feather_radius"])
            stage_mode = "full_run"
        else:
            raise ValueError(f"Unsupported staged compositor execution mode: {execution_mode!r}.")
        staged = _apply_staged_layer_options(
            staged,
            background_options["foreground_blend"],
            face_options["face_blend"],
            face_options["face_feather_radius"],
        )
        return _composite_staged_foregrounds(
            background, staged, placement_data, background_options["feather_radius"],
            stage_mode=stage_mode,
            image_resize_method=background_options["image_resize_method"],
            mask_resize_method=background_options["mask_resize_method"],
        )


class UC_StagedLayeredBackgroundComposite(io.ComfyNode):
    _staged_by_node = {}

    @classmethod
    def define_schema(cls):
        foreground_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("foreground", lazy=True, tooltip="Foreground image to isolate and retain as a placeable layer."),
            prefix="foreground_", min=1, max=50
        )
        return io.Schema(
            node_id="UC_StagedLayeredBackgroundComposite",
            display_name="Staged Background Composite",
            description=(
                "Run only this node with execution_mode set to run_staging to stage foreground objects for placement. "
                "After arranging them, use run_staged to composite the retained cutouts, or full_run to restage changed "
                "inputs and composite them in one execution."
            ),
            category="utils/image",
            inputs=[
                io.BackgroundRemoval.Input(
                    "background_removal_model",
                    display_name="background_removal_model_opt",
                    optional=True,
                    lazy=True,
                    tooltip=(
                        "Optional external Core background-removal model. When connected it overrides the internal "
                        "BiRefNet/Lucida selector."
                    ),
                ),
                io.Image.Input("background", tooltip="Single image used as the scene canvas."),
                io.Combo.Input(
                    "execution_mode",
                    options=["run_staging", "run_staged", "full_run"],
                    default="full_run",
                    tooltip=(
                        "run_staging: refresh retained cutouts and placement previews only. "
                        "run_staged: composite the retained cutouts without evaluating foreground inputs. "
                        "full_run: refresh cutouts from current inputs and composite them immediately."
                    ),
                ),
                io.Float.Input("mask_threshold", default=0.50, min=0.0, max=1.0, step=0.01, tooltip=_MASK_THRESHOLD_TOOLTIP),
                io.Int.Input("border_cleanup_width", default=2, min=0, max=64, step=1, advanced=True, tooltip=_BORDER_CLEANUP_TOOLTIP),
                io.Int.Input("artifact_cleanup_radius", default=2, min=0, max=64, step=1, advanced=True, tooltip=_ARTIFACT_CLEANUP_TOOLTIP),
                io.Int.Input("gap_fill_radius", default=2, min=0, max=64, step=1, advanced=True, tooltip=_GAP_FILL_TOOLTIP),
                io.Int.Input("feather_radius", default=2, min=0, max=64, step=1, advanced=True, tooltip=_FEATHER_TOOLTIP),
                io.Combo.Input("image_resize_method", options=_COMPOSITE_RESIZE_METHODS, default="auto", advanced=True, tooltip=_IMAGE_RESIZE_TOOLTIP),
                io.Combo.Input("mask_resize_method", options=_COMPOSITE_RESIZE_METHODS, default="auto", advanced=True, tooltip=_MASK_RESIZE_TOOLTIP),
                io.String.Input(
                    "placement_data",
                    default='{"version":2,"workspace_padding":0.5,"layers":{}}',
                    advanced=True,
                    tooltip="Versioned per-layer placement data managed by the LiteGraph scene editor.",
                ),
                io.Combo.Input(
                    "background_removal_model_name",
                    options=["birefnet", "lucida"],
                    default="birefnet",
                    tooltip=(
                        "Internal model used when background_removal_model_opt is disconnected. Requires the exact "
                        "checkpoint filename under models/background_removal."
                    ),
                ),
                io.Autogrow.Input(
                    "foreground_images",
                    template=foreground_template,
                    tooltip="Foregrounds staged and composited from foreground_0 at the back to the highest socket at the front.",
                ),
            ],
            outputs=[
                io.Image.Output("image"),
                io.Mask.Output("mask"),
                io.BoundingBox.Output("bounding_boxes", display_name="Boxes"),
                io.Mask.Output("layer_masks", display_name="Layer Masks"),
            ],
            hidden=[io.Hidden.unique_id],
            is_output_node=True,
        )

    @classmethod
    def check_lazy_status(
        cls,
        execution_mode,
        background_removal_model=_MISSING,
        foreground_images=None,
        **kwargs,
    ):
        if execution_mode is True or execution_mode == "run_staged":
            return []
        required = []
        if background_removal_model is None:
            required.append("background_removal_model")
        for value in (foreground_images or {}).values():
            if isinstance(value, tuple) and len(value) == 2:
                evaluated, original_key = value
            else:
                evaluated, original_key = value, None
            if evaluated is None and original_key:
                required.append(original_key)
        return required

    @classmethod
    def execute(
        cls,
        background,
        foreground_images,
        execution_mode,
        mask_threshold,
        border_cleanup_width,
        artifact_cleanup_radius,
        gap_fill_radius,
        placement_data,
        feather_radius,
        image_resize_method="auto",
        mask_resize_method="auto",
        background_removal_model_name="birefnet",
        background_removal_model=None,
    ):
        node_id = str(cls.hidden.unique_id or "")
        if isinstance(execution_mode, bool):
            execution_mode = "run_staged" if execution_mode else "run_staging"
        if execution_mode not in ("run_staging", "run_staged", "full_run"):
            raise ValueError(f"Unsupported staged compositor execution mode: {execution_mode!r}.")
        if execution_mode == "run_staged":
            if node_id not in cls._staged_by_node:
                raise ValueError(
                    "No retained foreground stage is available for this compositor. "
                    "Run once with execution_mode set to run_staging or full_run."
                )
            staged = cls._staged_by_node[node_id]
            stage_mode = "retained"
        else:
            if background_removal_model is None:
                background_removal_model = _load_internal_background_removal_model(
                    background_removal_model_name
                )
                effective_model_name = str(background_removal_model_name or "birefnet").lower()
            else:
                effective_model_name = "external"
            staged = _stage_layered_foregrounds(
                background_removal_model,
                foreground_images,
                mask_threshold,
                border_cleanup_width,
                artifact_cleanup_radius,
                gap_fill_radius,
                mask_resize_method,
                placement_data,
            )
            staged["background_removal_model_name"] = effective_model_name
            cls._staged_by_node[node_id] = staged
            if execution_mode == "run_staging":
                return _preview_staged_foregrounds(background, staged, feather_radius)
            stage_mode = "full_run"
        return _composite_staged_foregrounds(
            background,
            staged,
            placement_data,
            feather_radius,
            stage_mode=stage_mode,
            image_resize_method=image_resize_method,
            mask_resize_method=mask_resize_method,
        )


class UC_LayeredBackgroundComposite(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        foreground_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("foreground", tooltip="Foreground image to isolate and add as a placeable layer."),
            prefix="foreground_", min=1, max=50
        )
        return io.Schema(
            node_id="UC_LayeredBackgroundComposite",
            display_name="Layered Background Composite",
            category="utils/image",
            inputs=[
                io.BackgroundRemoval.Input(
                    "background_removal_model",
                    tooltip="Core background-removal model used to isolate every foreground layer.",
                ),
                io.Image.Input("background", tooltip="Single image used as the scene canvas."),
                io.Float.Input("mask_threshold", default=0.50, min=0.0, max=1.0, step=0.01, tooltip=_MASK_THRESHOLD_TOOLTIP),
                io.Int.Input("border_cleanup_width", default=2, min=0, max=64, step=1, advanced=True, tooltip=_BORDER_CLEANUP_TOOLTIP),
                io.Int.Input("artifact_cleanup_radius", default=2, min=0, max=64, step=1, advanced=True, tooltip=_ARTIFACT_CLEANUP_TOOLTIP),
                io.Int.Input("gap_fill_radius", default=2, min=0, max=64, step=1, advanced=True, tooltip=_GAP_FILL_TOOLTIP),
                io.Int.Input("feather_radius", default=2, min=0, max=64, step=1, advanced=True, tooltip=_FEATHER_TOOLTIP),
                io.Combo.Input("image_resize_method", options=_COMPOSITE_RESIZE_METHODS, default="auto", advanced=True, tooltip=_IMAGE_RESIZE_TOOLTIP),
                io.Combo.Input("mask_resize_method", options=_COMPOSITE_RESIZE_METHODS, default="auto", advanced=True, tooltip=_MASK_RESIZE_TOOLTIP),
                io.String.Input(
                    "placement_data",
                    default='{"version":2,"workspace_padding":0.5,"layers":{}}',
                    advanced=True,
                    tooltip="Versioned per-layer placement data managed by the LiteGraph scene editor.",
                ),
                io.Autogrow.Input(
                    "foreground_images",
                    template=foreground_template,
                    tooltip="One image per socket, composited from foreground_0 at the back to the highest socket at the front.",
                ),
            ],
            outputs=[io.Image.Output("image"), io.Mask.Output("mask")],
        )

    @classmethod
    def execute(
        cls,
        background_removal_model,
        background,
        foreground_images,
        placement_data,
        mask_threshold,
        border_cleanup_width,
        artifact_cleanup_radius,
        gap_fill_radius,
        feather_radius,
        image_resize_method="auto",
        mask_resize_method="auto",
    ):
        if not torch.is_tensor(background) or background.ndim != 4 or background.shape[0] != 1:
            raise ValueError("Layered Background Composite requires exactly one background image.")
        if background.shape[-1] < 3:
            raise ValueError("Background image must have at least three channels.")
        foregrounds = _ordered_single_foregrounds(foreground_images)
        if not foregrounds:
            raise ValueError("Layered Background Composite requires at least one foreground image.")
        placements = _parse_layer_placements(placement_data)
        placement_version, _, _, workspace_padding = _parse_layer_payload(placement_data)
        foreground_by_socket = dict(foregrounds)
        foregrounds = [
            (key, foreground_by_socket[key])
            for key in _ordered_layer_keys(placement_data, foreground_by_socket)
        ]

        scene = background[..., :3].clone()
        background_height, background_width = scene.shape[1:3]
        combined_mask = scene.new_zeros((1, background_height, background_width))
        layer_metadata = []

        for key, foreground in foregrounds:
            if foreground.shape[-1] < 3:
                raise ValueError(f"Foreground input {key} must have at least three channels.")
            foreground = foreground[..., :3]
            raw_mask = background_removal_model.encode_image(foreground)
            if not torch.is_tensor(raw_mask):
                raise ValueError(f"Background removal returned an invalid mask for {key}.")
            if raw_mask.ndim == 4 and raw_mask.shape[1] == 1:
                raw_mask = raw_mask[:, 0]
            elif raw_mask.ndim == 4 and raw_mask.shape[-1] == 1:
                raw_mask = raw_mask[..., 0]
            if raw_mask.ndim != 3 or raw_mask.shape[0] != 1:
                raise ValueError(f"Background removal must return one [batch, height, width] mask for {key}.")
            if raw_mask.shape[-2:] != foreground.shape[1:3]:
                raw_mask = _resize_composite_mask(raw_mask, foreground.shape[2], foreground.shape[1], mask_resize_method)

            refined = _refine_foreground_mask(
                raw_mask[0],
                float(mask_threshold),
                border_cleanup_width,
                artifact_cleanup_radius,
                gap_fill_radius,
            )
            points = torch.nonzero(refined > 0, as_tuple=False)
            if points.numel() == 0:
                raise ValueError(f"Background removal produced an empty foreground mask for {key}.")
            top = int(points[:, 0].min())
            bottom = int(points[:, 0].max()) + 1
            left = int(points[:, 1].min())
            right = int(points[:, 1].max()) + 1
            crop = foreground[:, top:bottom, left:right]
            crop_mask = refined[None, top:bottom, left:right]
            crop_height, crop_width = crop.shape[1:3]

            placement = placements.get(
                key,
                {**(_DEFAULT_LAYER_PLACEMENT_V2 if placement_version == 2 else _DEFAULT_LAYER_PLACEMENT), "_version": placement_version},
            )
            desired_flip = bool(placement.get("flip_horizontal", False))
            desired_flip_vertical = bool(placement.get("flip_vertical", False))
            if desired_flip:
                crop = torch.flip(crop, dims=(2,))
                crop_mask = torch.flip(crop_mask, dims=(2,))
            if desired_flip_vertical:
                crop = torch.flip(crop, dims=(1,))
                crop_mask = torch.flip(crop_mask, dims=(1,))
            target_longest = max(1, round(min(background_height, background_width) * placement["scale"]))
            scale = target_longest / max(crop_height, crop_width)
            placed_height = max(1, round(crop_height * scale))
            placed_width = max(1, round(crop_width * scale))
            resized_foreground = _resize_composite_image(crop, placed_width, placed_height, image_resize_method).to(scene)
            resized_mask = _resize_composite_mask(crop_mask, placed_width, placed_height, mask_resize_method).to(scene)
            resized_mask = resized_mask[0]
            alpha = _feather_mask(resized_mask, -int(feather_radius)) if feather_radius else resized_mask
            offset_x, offset_y = _placement_offsets(
                background_width,
                background_height,
                placed_width,
                placed_height,
                placement,
                workspace_padding if placement_version == 2 else 0.0,
            )

            slices = _visible_placement_slices(
                background_width, background_height, placed_width, placed_height, offset_x, offset_y
            )
            if slices is not None:
                (
                    destination_top, destination_bottom, destination_left, destination_right,
                    source_top, source_bottom, source_left, source_right,
                ) = slices
                placed_alpha = alpha[source_top:source_bottom, source_left:source_right]
                placed_foreground = resized_foreground[0, source_top:source_bottom, source_left:source_right]
                region = scene[0, destination_top:destination_bottom, destination_left:destination_right]
                scene[0, destination_top:destination_bottom, destination_left:destination_right] = (
                    region * (1.0 - placed_alpha.unsqueeze(-1)) + placed_foreground * placed_alpha.unsqueeze(-1)
                )
                mask_region = combined_mask[0, destination_top:destination_bottom, destination_left:destination_right]
                combined_mask[0, destination_top:destination_bottom, destination_left:destination_right] = (
                    mask_region + placed_alpha * (1.0 - mask_region)
                )

            preview_alpha = crop_mask[0]
            if feather_radius:
                preview_alpha = _feather_mask(preview_alpha, -int(feather_radius))
            preview_rgba = torch.cat((crop[0], preview_alpha.unsqueeze(-1)), dim=-1).unsqueeze(0)
            layer_metadata.append({
                "socket": key,
                "crop_width": crop_width,
                "crop_height": crop_height,
                "preview_tensor": preview_rgba,
                "flip_horizontal": desired_flip,
                "flip_vertical": desired_flip_vertical,
            })

        editor_metadata = {
            "version": 1,
            "background": {"width": background_width, "height": background_height},
            "layers": [],
        }
        for layer in layer_metadata:
            entry = {
                "socket": layer["socket"],
                "crop_width": layer["crop_width"],
                "crop_height": layer["crop_height"],
                "flip_horizontal": layer["flip_horizontal"],
                "flip_vertical": layer["flip_vertical"],
            }
            try:
                entry["preview"] = _save_editor_preview(
                    layer["preview_tensor"], f"UC_layered_{layer['socket']}"
                )
            except Exception:
                logging.warning("Unable to create editor cutout preview for %s.", layer["socket"], exc_info=True)
            editor_metadata["layers"].append(entry)

        return io.NodeOutput(scene, combined_mask, ui={"uc_layered_scene_editor": [editor_metadata]})


class UC_MediaPipeFaceCompositeOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_MediaPipeFaceCompositeOptions",
            display_name="MediaPipe Face Composite Options",
            category="utils/image",
            inputs=[
                io.Int.Input("bbox_expansion", default=64, min=0, max=MAX_RESOLUTION, step=1),
                io.Int.Input("mask_expansion", default=0, min=-MAX_RESOLUTION, max=MAX_RESOLUTION, step=1),
                io.Int.Input("feather_radius", default=8, min=-512, max=512, step=1),
                io.Float.Input("target_warp_strength", default=1.0, min=0.0, max=2.0, step=0.01),
                io.Int.Input("warp_decay_radius", default=64, min=1, max=MAX_RESOLUTION, step=1),
                io.Float.Input("score_thresh", default=0.25, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[FaceCompositeOptionsType.Output()],
        )

    @classmethod
    def execute(cls, bbox_expansion, mask_expansion, feather_radius, target_warp_strength, warp_decay_radius, score_thresh):
        return io.NodeOutput({
            "bbox_expansion": int(bbox_expansion),
            "mask_expansion": int(mask_expansion),
            "feather_radius": int(feather_radius),
            "target_warp_strength": float(target_warp_strength),
            "warp_decay_radius": int(warp_decay_radius),
            "score_thresh": float(score_thresh),
        })


class UC_MediaPipeFaceComposite(io.ComfyNode):
    DEFAULT_OPTIONS = {
        "bbox_expansion": 64,
        "mask_expansion": 0,
        "feather_radius": 8,
        "target_warp_strength": 1.0,
        "warp_decay_radius": 64,
        "score_thresh": 0.25,
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_MediaPipeFaceComposite",
            display_name="MediaPipe Face Composite",
            category="utils/image",
            description="Composites the largest source face into the largest target face using full-range MediaPipe detection.",
            inputs=[
                FaceDetectionType.Input("face_detection_model"),
                io.BackgroundRemoval.Input("background_removal_model"),
                io.Image.Input("source"),
                io.Image.Input("target"),
                FaceCompositeOptionsType.Input("options", optional=True),
            ],
            outputs=[io.Image.Output("image"), io.Image.Output("face_crop", display_name="Face Crop")],
        )

    @classmethod
    def execute(cls, face_detection_model, background_removal_model, source, target, options=None):
        if source.shape[0] != 1 or target.shape[0] != 1:
            raise ValueError("MediaPipe Face Composite currently requires one source and one target image.")
        options = cls.DEFAULT_OPTIONS | (options or {})
        score_thresh = options["score_thresh"]
        source = source[..., :3]
        target = target[..., :3]
        source_uint8 = source.mul(255.0).add(0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
        target_uint8 = target.mul(255.0).add(0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
        source_face = _largest_face(face_detection_model.detect_batch([source_uint8], num_faces=1, score_thresh=score_thresh, variant="full")[0], "source")
        target_face = _largest_face(face_detection_model.detect_batch([target_uint8], num_faces=1, score_thresh=score_thresh, variant="full")[0], "target")
        ring = _ordered_ring(face_detection_model.connection_sets["face_oval"])

        source = source.to(target)
        source_points = source_face["landmarks_xy"][ring]
        target_points = target_face["landmarks_xy"][ring]
        source_mask = _polygon_mask(source.shape[1], source.shape[2], source_points, target.device, target.dtype)
        foreground = background_removal_model.encode_image(source)
        if foreground.shape[-2:] != source.shape[1:3]:
            foreground = _resize_mask(foreground, source.shape[2], source.shape[1], "bilinear")
        foreground = foreground[0].to(target).clamp(0.0, 1.0)

        padding = options["bbox_expansion"]
        sx1, sy1, sx2, sy2 = _expanded_box(source_face["bbox_xyxy"], padding, source.shape[2], source.shape[1])
        tx1, ty1, tx2, ty2 = _expanded_box(target_face["bbox_xyxy"], padding, target.shape[2], target.shape[1])
        source_crop = source[0, sy1:sy2, sx1:sx2]
        source_oval = source_mask[sy1:sy2, sx1:sx2]
        source_foreground = foreground[sy1:sy2, sx1:sx2]
        target_crop = target[0, ty1:ty2, tx1:tx2]

        local_source_points = source_points - np.array([sx1, sy1], dtype=np.float32)
        local_target_points = target_points - np.array([tx1, ty1], dtype=np.float32)
        scale, rotation, translation = _similarity_transform(local_source_points, local_target_points)
        placed_source, placed_oval, placed_foreground = _transform_source(
            source_crop,
            source_oval,
            source_foreground,
            target_crop.shape[0],
            target_crop.shape[1],
            scale,
            rotation,
            translation,
        )
        placed_source_points = scale * (local_source_points @ rotation.T) + translation
        warped_target = _warp_target(target_crop, placed_source_points, local_target_points, options["target_warp_strength"], options["warp_decay_radius"])

        opaque = _expand_mask(placed_oval, options["mask_expansion"]).clamp(0.0, 1.0)
        inverted_foreground = 1.0 - placed_foreground
        solid_foreground = ((placed_foreground - inverted_foreground) * 2.0).clamp(0.0, 1.0)
        alpha = _feather_mask(opaque, options["feather_radius"]) * solid_foreground
        completed_crop = warped_target * (1.0 - alpha.unsqueeze(-1)) + placed_source * alpha.unsqueeze(-1)
        result = target.clone()
        result[0, ty1:ty2, tx1:tx2] = completed_crop
        return io.NodeOutput(result, completed_crop.unsqueeze(0))
