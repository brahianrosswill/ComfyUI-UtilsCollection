import json
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from comfy.utils import common_upscale
from comfy_api.latest import io, ui
from nodes import MAX_RESOLUTION


FaceDetectionType = io.Custom("FACE_DETECTION_MODEL")
FaceCompositeOptionsType = io.Custom("UC_FACE_COMPOSITE_OPTIONS")
LayeredForegroundStageType = io.Custom("UC_LAYERED_FOREGROUND_STAGE")

_RESIZE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
_DEFAULT_LAYER_PLACEMENT = {
    "scale": 0.9,
    "long_axis_shift": 0.0,
    "short_axis_shift": 0.0,
}


def _resize_image(image, width, height, method, crop="disabled"):
    return common_upscale(image.movedim(-1, 1), width, height, method, crop).movedim(1, -1)


def _resize_mask(mask, width, height, method, crop="disabled"):
    mask = mask.unsqueeze(1)
    if method == "lanczos":
        mask = common_upscale(mask.repeat(1, 3, 1, 1), width, height, method, crop)[:, :1]
    else:
        mask = common_upscale(mask, width, height, method, crop)
    return mask.squeeze(1)


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
    blurred = _blur_mask(mask.unsqueeze(0), abs(int(radius)))[0].clamp(0.0, 1.0)
    if radius > 0:
        return torch.maximum(mask, blurred)
    return torch.minimum(mask, blurred)


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


def _parse_layer_placements(value):
    if value is None or value == "":
        value = {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"Layer placement data is not valid JSON: {error.msg}.") from error
    if not isinstance(value, dict):
        raise ValueError("Layer placement data must be a JSON object.")
    version = value.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported layer placement data version: {version}.")
    layers = value.get("layers", {})
    if not isinstance(layers, dict):
        raise ValueError("Layer placement data 'layers' must be a JSON object.")

    parsed = {}
    for key, placement in layers.items():
        if not isinstance(key, str) or not isinstance(placement, dict):
            raise ValueError("Every layer placement must be an object keyed by its foreground socket name.")
        result = dict(_DEFAULT_LAYER_PLACEMENT)
        for field, minimum, maximum in (
            ("scale", 0.05, 10.0),
            ("long_axis_shift", -1.0, 1.0),
            ("short_axis_shift", -1.0, 1.0),
        ):
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
        parsed[key] = result
    return parsed


def _placement_offsets(background_width, background_height, placed_width, placed_height, placement):
    long_shift = (placement["long_axis_shift"] + 1.0) / 2.0
    short_shift = (placement["short_axis_shift"] + 1.0) / 2.0
    if background_width > background_height:
        offset_x = round((background_width - placed_width) * long_shift)
        offset_y = round((background_height - placed_height) * short_shift)
    elif background_height > background_width:
        offset_y = round((background_height - placed_height) * long_shift)
        offset_x = round((background_width - placed_width) * short_shift)
    else:
        offset_x = round((background_width - placed_width) * long_shift)
        offset_y = round((background_height - placed_height) * short_shift)
    return offset_x, offset_y


def _bounded_preview(image, longest):
    height, width = image.shape[1:3]
    if max(height, width) <= longest:
        return image
    ratio = float(longest) / max(height, width)
    return _resize_image(image, max(1, round(width * ratio)), max(1, round(height * ratio)), "bicubic")


def _save_editor_preview(image, prefix, longest):
    preview = _bounded_preview(image, longest)
    saved = ui.ImageSaveHelper.save_images(
        preview,
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
                io.Int.Output("crop_x"),
                io.Int.Output("crop_y"),
                io.Int.Output("crop_width"),
                io.Int.Output("crop_height"),
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
                io.Autogrow.Input("foreground_images", template=foreground_template, tooltip="Images to isolate, resize, center, and composite in socket order."),
                io.Float.Input("foreground_scale", default=0.90, min=0.05, max=10.0, step=0.01, tooltip="Fraction of the background's shortest side occupied by the foreground's longest bound. Values above 1 overscale and crop at the canvas edges."),
                io.Float.Input("long_axis_shift", default=0.0, min=-1.0, max=1.0, step=0.01, tooltip="Position along the background's longest axis: -1 is left/up, 0 is centered, and 1 is right/down."),
                io.Float.Input("short_axis_shift", default=0.0, min=-1.0, max=1.0, step=0.01, tooltip="Position along the background's shortest axis: -1 is up/left, 0 is centered, and 1 is down/right."),
                io.Float.Input("mask_threshold", default=0.50, min=0.0, max=1.0, step=0.01, tooltip="Minimum model confidence retained as solid foreground."),
                io.Int.Input("border_cleanup_width", default=2, min=0, max=64, step=1, advanced=True, tooltip="Width of the source-edge strip where weak foreground predictions are removed."),
                io.Int.Input("artifact_cleanup_radius", default=2, min=0, max=64, step=1, advanced=True, tooltip="Opening radius used to remove small and thin mask artifacts."),
                io.Int.Input("gap_fill_radius", default=2, min=0, max=64, step=1, advanced=True, tooltip="Closing radius used to fill small cracks and holes in the foreground."),
                io.Int.Input("feather_radius", default=2, min=0, max=64, step=1, advanced=True, tooltip="Inward edge softness; the foreground interior remains fully opaque."),
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
    ):
        if not torch.is_tensor(background) or background.ndim != 4 or background.shape[0] != 1:
            raise ValueError("Unified Background Replace requires exactly one background image.")
        if background.shape[-1] < 3:
            raise ValueError("Background image must have at least three channels.")
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
                raw_mask = _resize_mask(raw_mask, foreground.shape[2], foreground.shape[1], "bilinear")
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
            resized_foreground = _resize_image(crop, placed_width, placed_height, "bicubic").to(background)
            resized_mask = _resize_mask(crop_mask, placed_width, placed_height, "nearest-exact").to(background)
            resized_mask = (resized_mask[0] >= 0.5).to(background)
            alpha = _feather_mask(resized_mask, -int(feather_radius)) if feather_radius else resized_mask

            offset_y = (background_height - placed_height) // 2
            offset_x = (background_width - placed_width) // 2
            long_shift = (float(long_axis_shift) + 1.0) / 2.0
            short_shift = (float(short_axis_shift) + 1.0) / 2.0
            if background_width > background_height:
                offset_x = round((background_width - placed_width) * long_shift)
                offset_y = round((background_height - placed_height) * short_shift)
            elif background_height > background_width:
                offset_y = round((background_height - placed_height) * long_shift)
                offset_x = round((background_width - placed_width) * short_shift)
            else:
                # A square canvas has no intrinsic long or short axis. Keep both
                # controls useful by mapping long to horizontal and short to vertical.
                offset_x = round((background_width - placed_width) * long_shift)
                offset_y = round((background_height - placed_height) * short_shift)
            destination_top = max(0, offset_y)
            destination_bottom = min(background_height, offset_y + placed_height)
            destination_left = max(0, offset_x)
            destination_right = min(background_width, offset_x + placed_width)
            source_top = destination_top - offset_y
            source_bottom = source_top + destination_bottom - destination_top
            source_left = destination_left - offset_x
            source_right = source_left + destination_right - destination_left
            composite = background.clone()
            placed_alpha = alpha[source_top:source_bottom, source_left:source_right]
            placed_foreground = resized_foreground[0, source_top:source_bottom, source_left:source_right]
            region = composite[0, destination_top:destination_bottom, destination_left:destination_right]
            composite[0, destination_top:destination_bottom, destination_left:destination_right] = (
                region * (1.0 - placed_alpha.unsqueeze(-1)) + placed_foreground * placed_alpha.unsqueeze(-1)
            )
            canvas_mask = background.new_zeros((1, background_height, background_width))
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
):
    foregrounds = _ordered_single_foregrounds(foreground_images)
    if not foregrounds:
        raise ValueError("Layered foreground staging requires at least one foreground image.")
    layers = []
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
            raw_mask = _resize_mask(raw_mask, foreground.shape[2], foreground.shape[1], "bilinear")
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
        layers.append({
            "socket": key,
            "image": foreground[:, top:bottom, left:right],
            "mask": refined[None, top:bottom, left:right],
        })
    return {"version": 1, "layers": layers}


def _composite_staged_foregrounds(background, staged_foregrounds, placement_data, feather_radius):
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
    scene = background[..., :3].clone()
    background_height, background_width = scene.shape[1:3]
    combined_mask = scene.new_zeros((1, background_height, background_width))
    editor_layers = []

    for layer in layers:
        key = layer["socket"]
        crop = layer["image"]
        crop_mask = layer["mask"]
        crop_height, crop_width = crop.shape[1:3]
        placement = placements.get(key, _DEFAULT_LAYER_PLACEMENT)
        target_longest = max(1, round(min(background_height, background_width) * placement["scale"]))
        scale = target_longest / max(crop_height, crop_width)
        placed_height = max(1, round(crop_height * scale))
        placed_width = max(1, round(crop_width * scale))
        resized_foreground = _resize_image(crop, placed_width, placed_height, "bicubic").to(scene)
        resized_mask = _resize_mask(crop_mask, placed_width, placed_height, "nearest-exact").to(scene)
        resized_mask = (resized_mask[0] >= 0.5).to(scene)
        alpha = _feather_mask(resized_mask, -int(feather_radius)) if feather_radius else resized_mask
        offset_x, offset_y = _placement_offsets(
            background_width, background_height, placed_width, placed_height, placement
        )
        destination_top = max(0, offset_y)
        destination_bottom = min(background_height, offset_y + placed_height)
        destination_left = max(0, offset_x)
        destination_right = min(background_width, offset_x + placed_width)
        source_top = destination_top - offset_y
        source_bottom = source_top + destination_bottom - destination_top
        source_left = destination_left - offset_x
        source_right = source_left + destination_right - destination_left
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
        editor_layers.append({
            "socket": key,
            "crop_width": crop_width,
            "crop_height": crop_height,
            "preview_tensor": torch.cat((crop[0], preview_alpha.unsqueeze(-1)), dim=-1).unsqueeze(0),
        })

    editor_metadata = {
        "version": 1,
        "background": {"width": background_width, "height": background_height},
        "layers": [],
    }
    try:
        editor_metadata["background"]["preview"] = _save_editor_preview(
            background[..., :3], "UC_layered_background", 1024
        )
    except Exception:
        logging.warning("Unable to create staged layered-composite background preview.", exc_info=True)
    for layer in editor_layers:
        entry = {key: layer[key] for key in ("socket", "crop_width", "crop_height")}
        try:
            entry["preview"] = _save_editor_preview(
                layer["preview_tensor"], f"UC_layered_{layer['socket']}", 512
            )
        except Exception:
            logging.warning("Unable to create staged editor cutout preview for %s.", layer["socket"], exc_info=True)
        editor_metadata["layers"].append(entry)
    return io.NodeOutput(scene, combined_mask, ui={"uc_layered_scene_editor": [editor_metadata]})


class UC_LayeredForegroundStage(io.ComfyNode):
    _staged_by_node = {}

    @classmethod
    def define_schema(cls):
        foreground_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("foreground"), prefix="foreground_", min=1, max=50
        )
        return io.Schema(
            node_id="UC_LayeredForegroundStage",
            display_name="Layered Foreground Stage (Experimental)",
            category="utils/image",
            inputs=[
                io.BackgroundRemoval.Input("background_removal_model"),
                io.Autogrow.Input("foreground_images", template=foreground_template),
                io.Boolean.Input(
                    "use_staged",
                    default=False,
                    tooltip="Reuse this node's last successful cutouts and ignore changed foreground values.",
                ),
                io.Float.Input("mask_threshold", default=0.50, min=0.0, max=1.0, step=0.01),
                io.Int.Input("border_cleanup_width", default=2, min=0, max=64, step=1, advanced=True),
                io.Int.Input("artifact_cleanup_radius", default=2, min=0, max=64, step=1, advanced=True),
                io.Int.Input("gap_fill_radius", default=2, min=0, max=64, step=1, advanced=True),
            ],
            outputs=[LayeredForegroundStageType.Output("staged_foregrounds")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        background_removal_model,
        foreground_images,
        use_staged,
        mask_threshold,
        border_cleanup_width,
        artifact_cleanup_radius,
        gap_fill_radius,
    ):
        node_id = str(cls.hidden.unique_id or "")
        if use_staged:
            if node_id not in cls._staged_by_node:
                raise ValueError("No staged foregrounds are available. Run once with use_staged disabled.")
            return io.NodeOutput(cls._staged_by_node[node_id])
        staged = _stage_layered_foregrounds(
            background_removal_model,
            foreground_images,
            mask_threshold,
            border_cleanup_width,
            artifact_cleanup_radius,
            gap_fill_radius,
        )
        cls._staged_by_node[node_id] = staged
        return io.NodeOutput(staged)


class UC_StagedLayeredBackgroundComposite(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_StagedLayeredBackgroundComposite",
            display_name="Staged Layered Background Composite (Experimental)",
            category="utils/image",
            inputs=[
                io.Image.Input("background", tooltip="Single image used as the scene canvas."),
                LayeredForegroundStageType.Input("staged_foregrounds"),
                io.String.Input(
                    "placement_data",
                    default='{"version":1,"layers":{}}',
                    advanced=True,
                    tooltip="Versioned per-layer placement data managed by the LiteGraph scene editor.",
                ),
                io.Int.Input("feather_radius", default=2, min=0, max=64, step=1, advanced=True),
            ],
            outputs=[io.Image.Output("image"), io.Mask.Output("mask")],
        )

    @classmethod
    def execute(cls, background, staged_foregrounds, placement_data, feather_radius):
        return _composite_staged_foregrounds(background, staged_foregrounds, placement_data, feather_radius)


class UC_LayeredBackgroundComposite(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        foreground_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("foreground"), prefix="foreground_", min=1, max=50
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
                io.Autogrow.Input(
                    "foreground_images",
                    template=foreground_template,
                    tooltip="One image per socket, composited from foreground_0 at the back to the highest socket at the front.",
                ),
                io.String.Input(
                    "placement_data",
                    default='{"version":1,"layers":{}}',
                    advanced=True,
                    tooltip="Versioned per-layer placement data managed by the LiteGraph scene editor.",
                ),
                io.Float.Input("mask_threshold", default=0.50, min=0.0, max=1.0, step=0.01),
                io.Int.Input("border_cleanup_width", default=2, min=0, max=64, step=1, advanced=True),
                io.Int.Input("artifact_cleanup_radius", default=2, min=0, max=64, step=1, advanced=True),
                io.Int.Input("gap_fill_radius", default=2, min=0, max=64, step=1, advanced=True),
                io.Int.Input("feather_radius", default=2, min=0, max=64, step=1, advanced=True),
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
    ):
        if not torch.is_tensor(background) or background.ndim != 4 or background.shape[0] != 1:
            raise ValueError("Layered Background Composite requires exactly one background image.")
        if background.shape[-1] < 3:
            raise ValueError("Background image must have at least three channels.")
        foregrounds = _ordered_single_foregrounds(foreground_images)
        if not foregrounds:
            raise ValueError("Layered Background Composite requires at least one foreground image.")
        placements = _parse_layer_placements(placement_data)

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
                raw_mask = _resize_mask(raw_mask, foreground.shape[2], foreground.shape[1], "bilinear")

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

            placement = placements.get(key, _DEFAULT_LAYER_PLACEMENT)
            target_longest = max(1, round(min(background_height, background_width) * placement["scale"]))
            scale = target_longest / max(crop_height, crop_width)
            placed_height = max(1, round(crop_height * scale))
            placed_width = max(1, round(crop_width * scale))
            resized_foreground = _resize_image(crop, placed_width, placed_height, "bicubic").to(scene)
            resized_mask = _resize_mask(crop_mask, placed_width, placed_height, "nearest-exact").to(scene)
            resized_mask = (resized_mask[0] >= 0.5).to(scene)
            alpha = _feather_mask(resized_mask, -int(feather_radius)) if feather_radius else resized_mask
            offset_x, offset_y = _placement_offsets(
                background_width,
                background_height,
                placed_width,
                placed_height,
                placement,
            )

            destination_top = max(0, offset_y)
            destination_bottom = min(background_height, offset_y + placed_height)
            destination_left = max(0, offset_x)
            destination_right = min(background_width, offset_x + placed_width)
            source_top = destination_top - offset_y
            source_bottom = source_top + destination_bottom - destination_top
            source_left = destination_left - offset_x
            source_right = source_left + destination_right - destination_left
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
            })

        editor_metadata = {
            "version": 1,
            "background": {"width": background_width, "height": background_height},
            "layers": [],
        }
        try:
            editor_metadata["background"]["preview"] = _save_editor_preview(
                background[..., :3], "UC_layered_background", 1024
            )
        except Exception:
            logging.warning("Unable to create layered-composite background editor preview.", exc_info=True)
        for layer in layer_metadata:
            entry = {
                "socket": layer["socket"],
                "crop_width": layer["crop_width"],
                "crop_height": layer["crop_height"],
            }
            try:
                entry["preview"] = _save_editor_preview(
                    layer["preview_tensor"], f"UC_layered_{layer['socket']}", 512
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
            outputs=[io.Image.Output("image"), io.Image.Output("face_crop")],
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
