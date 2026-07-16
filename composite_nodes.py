import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from comfy.utils import common_upscale
from comfy_api.latest import io
from nodes import MAX_RESOLUTION


FaceDetectionType = io.Custom("FACE_DETECTION_MODEL")
FaceCompositeOptionsType = io.Custom("UC_FACE_COMPOSITE_OPTIONS")

_RESIZE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]


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
    def execute(cls, image, mask, padding):
        if mask.shape[-2:] != image.shape[1:3]:
            mask = _resize_mask(mask, image.shape[2], image.shape[1], "nearest-exact")
        mask = _broadcast_batch(mask, image.shape[0], "Mask")
        x, y, width, height = _crop_bounds(mask, int(padding))
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


def _mask_extents(mask):
    points = torch.nonzero(mask > 0.5, as_tuple=False)
    if points.numel() == 0:
        raise ValueError("Face landmark mask is empty.")
    y = points[:, 0].to(torch.float32)
    x = points[:, 1].to(torch.float32)
    center_x = x.mean().item()
    center_y = y.mean().item()
    return center_x, center_y, center_x - x.min().item(), x.max().item() - center_x, center_y - y.min().item(), y.max().item() - center_y


def _place(value, target_height, target_width, x, y):
    if value.ndim == 3:
        output = value.new_zeros((target_height, target_width, value.shape[-1]))
    else:
        output = value.new_zeros((target_height, target_width))
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(target_width, x + value.shape[1])
    y2 = min(target_height, y + value.shape[0])
    if x2 > x1 and y2 > y1:
        output[y1:y2, x1:x2] = value[y1 - y:y2 - y, x1 - x:x2 - x]
    return output


def _smoothstep(value):
    value = value.clamp(0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _warp_target(target, source_oval_points, target_oval_points, strength, decay_radius):
    if strength <= 0:
        return target
    device = target.device
    dtype = target.dtype
    source_points = torch.as_tensor(source_oval_points, device=device, dtype=dtype)
    target_points = torch.as_tensor(target_oval_points, device=device, dtype=dtype)
    center = source_points.mean(dim=0)
    vectors = source_points - center
    angles = torch.atan2(vectors[:, 1], vectors[:, 0])
    order = torch.argsort(angles)
    angles = angles[order]
    radii = vectors.norm(dim=1)[order]
    deltas = (target_points - source_points)[order]
    angles = torch.cat((angles, angles[:1] + 2.0 * math.pi))
    radii = torch.cat((radii, radii[:1]))
    deltas = torch.cat((deltas, deltas[:1]), dim=0)

    height, width = target.shape[:2]
    yy, xx = torch.meshgrid(torch.arange(height, device=device, dtype=dtype), torch.arange(width, device=device, dtype=dtype), indexing="ij")
    dx = xx - center[0]
    dy = yy - center[1]
    pixel_angles = torch.atan2(dy, dx)
    pixel_angles = torch.where(pixel_angles < angles[0], pixel_angles + 2.0 * math.pi, pixel_angles)
    upper = torch.searchsorted(angles, pixel_angles, right=True).clamp(1, angles.shape[0] - 1)
    lower = upper - 1
    amount = (pixel_angles - angles[lower]) / (angles[upper] - angles[lower]).clamp_min(torch.finfo(dtype).eps)
    boundary_radius = radii[lower] + (radii[upper] - radii[lower]) * amount
    displacement = deltas[lower] + (deltas[upper] - deltas[lower]) * amount.unsqueeze(-1)
    radius = torch.sqrt(dx * dx + dy * dy)
    inside = _smoothstep(radius / boundary_radius.clamp_min(1.0))
    outside = 1.0 - _smoothstep((radius - boundary_radius) / max(float(decay_radius), 1.0))
    weight = torch.where(radius <= boundary_radius, inside, outside)
    edge_distance = torch.minimum(torch.minimum(xx, width - 1 - xx), torch.minimum(yy, height - 1 - yy))
    weight = weight * _smoothstep(edge_distance / max(float(decay_radius) * 0.5, 1.0)) * float(strength)
    sample_x = xx + displacement[..., 0] * weight
    sample_y = yy + displacement[..., 1] * weight
    grid_x = (sample_x + 0.5) * (2.0 / width) - 1.0
    grid_y = (sample_y + 0.5) * (2.0 / height) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    warped = F.grid_sample(target.movedim(-1, 0).unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False)
    return warped.squeeze(0).movedim(0, -1)


class UC_MediaPipeFaceCompositeOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_MediaPipeFaceCompositeOptions",
            display_name="MediaPipe Face Composite Options",
            category="utils/image",
            inputs=[
                io.Int.Input("bbox_expansion", default=64, min=0, max=MAX_RESOLUTION, step=1),
                io.Int.Input("mask_expansion", default=0, min=0, max=MAX_RESOLUTION, step=1),
                io.Int.Input("feather_radius", default=8, min=0, max=512, step=1),
                io.Float.Input("target_warp_strength", default=1.0, min=0.0, max=2.0, step=0.01),
                io.Int.Input("warp_decay_radius", default=64, min=1, max=MAX_RESOLUTION, step=1),
            ],
            outputs=[FaceCompositeOptionsType.Output()],
        )

    @classmethod
    def execute(cls, bbox_expansion, mask_expansion, feather_radius, target_warp_strength, warp_decay_radius):
        return io.NodeOutput({
            "bbox_expansion": int(bbox_expansion),
            "mask_expansion": int(mask_expansion),
            "feather_radius": int(feather_radius),
            "target_warp_strength": float(target_warp_strength),
            "warp_decay_radius": int(warp_decay_radius),
        })


class UC_MediaPipeFaceComposite(io.ComfyNode):
    DEFAULT_OPTIONS = {
        "bbox_expansion": 64,
        "mask_expansion": 0,
        "feather_radius": 8,
        "target_warp_strength": 1.0,
        "warp_decay_radius": 64,
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
        source = source[..., :3]
        target = target[..., :3]
        source_uint8 = source.mul(255.0).add(0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
        target_uint8 = target.mul(255.0).add(0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
        source_face = _largest_face(face_detection_model.detect_batch([source_uint8], num_faces=0, score_thresh=0.5, variant="full")[0], "source")
        target_face = _largest_face(face_detection_model.detect_batch([target_uint8], num_faces=0, score_thresh=0.5, variant="full")[0], "target")
        ring = _ordered_ring(face_detection_model.connection_sets["face_oval"])

        source = source.to(target)
        source_points = source_face["landmarks_xy"][ring]
        target_points = target_face["landmarks_xy"][ring]
        source_mask = _polygon_mask(source.shape[1], source.shape[2], source_points, target.device, target.dtype)
        target_mask = _polygon_mask(target.shape[1], target.shape[2], target_points, target.device, target.dtype)
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
        target_oval = target_mask[ty1:ty2, tx1:tx2]

        source_geometry = _mask_extents(source_oval)
        target_geometry = _mask_extents(target_oval)
        scale = max(target_geometry[index] / max(source_geometry[index], 1.0) for index in range(2, 6))
        scaled_width = max(1, round(source_crop.shape[1] * scale))
        scaled_height = max(1, round(source_crop.shape[0] * scale))
        source_crop = _resize_image(source_crop.unsqueeze(0), scaled_width, scaled_height, "lanczos")[0]
        source_oval = _resize_mask(source_oval.unsqueeze(0), scaled_width, scaled_height, "bilinear")[0]
        source_foreground = _resize_mask(source_foreground.unsqueeze(0), scaled_width, scaled_height, "bilinear")[0]
        source_center_x = source_geometry[0] * scale
        source_center_y = source_geometry[1] * scale
        place_x = round(target_geometry[0] - source_center_x)
        place_y = round(target_geometry[1] - source_center_y)
        placed_source = _place(source_crop, target_crop.shape[0], target_crop.shape[1], place_x, place_y)
        placed_oval = _place(source_oval, target_crop.shape[0], target_crop.shape[1], place_x, place_y)
        placed_foreground = _place(source_foreground, target_crop.shape[0], target_crop.shape[1], place_x, place_y)

        placed_source_points = (source_points - np.array([sx1, sy1], dtype=np.float32)) * scale + np.array([place_x, place_y], dtype=np.float32)
        local_target_points = target_points - np.array([tx1, ty1], dtype=np.float32)
        warped_target = _warp_target(target_crop, placed_source_points, local_target_points, options["target_warp_strength"], options["warp_decay_radius"])

        mask_expansion = options["mask_expansion"]
        opaque = placed_oval.unsqueeze(0).unsqueeze(0)
        if mask_expansion > 0:
            kernel = 2 * mask_expansion + 1
            opaque = F.max_pool2d(opaque, kernel, stride=1, padding=mask_expansion)
        opaque = opaque.squeeze(0).squeeze(0).clamp(0.0, 1.0)
        feathered = _blur_mask(opaque.unsqueeze(0), options["feather_radius"])[0].clamp(0.0, 1.0)
        alpha = torch.maximum(placed_oval, feathered) * placed_foreground
        completed_crop = warped_target * (1.0 - alpha.unsqueeze(-1)) + placed_source * alpha.unsqueeze(-1)
        result = target.clone()
        result[0, ty1:ty2, tx1:tx2] = completed_crop
        return io.NodeOutput(result, completed_crop.unsqueeze(0))
