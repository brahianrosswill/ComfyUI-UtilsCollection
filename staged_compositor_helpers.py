from collections import OrderedDict
from collections.abc import MutableMapping
import logging
import os

import folder_paths
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from comfy_api.latest import io

from .composite_helpers import (
    _DEFAULT_LAYER_PLACEMENT,
    _DEFAULT_LAYER_PLACEMENT_V2,
    _PAINT_LAYER_KEY,
    _feather_mask,
    _ordered_layer_keys,
    _ordered_single_foregrounds,
    _parse_layer_payload,
    _parse_layer_placements,
    _parse_paint_layer,
    _placement_offsets,
    _refine_foreground_mask,
    _resize_composite_image,
    _resize_composite_mask,
    _save_editor_preview,
    _visible_placement_slices,
)


_DEFAULT_CORNERS = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))


def resize_paint_rgba(rgba, width, height):
    """Resize straight RGBA through premultiplied alpha to avoid color fringes."""
    if rgba.ndim != 4 or rgba.shape[0] != 1 or rgba.shape[-1] != 4:
        raise ValueError("The staged paint asset must contain one RGBA image.")
    if rgba.shape[1:3] == (height, width):
        return rgba
    alpha = rgba[..., 3:4].clamp(0.0, 1.0)
    premultiplied = torch.cat((rgba[..., :3].clamp(0.0, 1.0) * alpha, alpha), dim=-1)
    resized = F.interpolate(
        premultiplied.movedim(-1, 1),
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    ).movedim(1, -1)
    resized_alpha = resized[..., 3:4].clamp(0.0, 1.0)
    resized_rgb = torch.where(
        resized_alpha > 1e-8,
        resized[..., :3] / resized_alpha.clamp_min(1e-8),
        torch.zeros_like(resized[..., :3]),
    ).clamp(0.0, 1.0)
    return torch.cat((resized_rgb, resized_alpha), dim=-1)


def load_staged_paint_rgba(paint, width, height, device, dtype):
    asset = paint["asset"]
    relative = os.path.join(asset["subfolder"], asset["filename"]).replace("\\", "/")
    annotated = f"{relative} [input]"
    if not folder_paths.exists_annotated_filepath(annotated):
        raise ValueError("The staged paint PNG is missing from ComfyUI input storage.")
    path = folder_paths.get_annotated_filepath(annotated)
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or int(getattr(image, "n_frames", 1)) != 1:
                raise ValueError("The staged paint asset must be one RGBA PNG image.")
            if image.mode != "RGBA":
                raise ValueError(
                    "The staged paint asset must preserve an RGBA channel."
                )
            pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("The staged paint PNG could not be decoded.") from error
    rgba = torch.from_numpy(pixels).unsqueeze(0).to(device=device, dtype=dtype)
    return resize_paint_rgba(rgba, width, height)


def paint_alpha_bounds(alpha):
    points = torch.nonzero(alpha > 0, as_tuple=False)
    if points.numel() == 0:
        return {"x": 0, "y": 0, "width": 0, "height": 0}
    top = int(points[:, 0].min())
    bottom = int(points[:, 0].max()) + 1
    left = int(points[:, 1].min())
    right = int(points[:, 1].max()) + 1
    return {"x": left, "y": top, "width": right - left, "height": bottom - top}


def _stored_tensor(tensor):
    import comfy.model_management

    stored = (
        tensor.detach()
        .to(
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        )
        .contiguous()
    )
    if stored.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr():
        stored = stored.clone()
    return stored


class RetainedStageCache(MutableMapping):
    def __init__(self, max_entries=8):
        self.max_entries = max(1, int(max_entries))
        self._entries = OrderedDict()

    def __getitem__(self, key):
        value = self._entries.pop(key)
        self._entries[key] = value
        return value

    def __setitem__(self, key, value):
        layers = [
            {
                **layer,
                "image": _stored_tensor(layer["image"]),
                "mask": _stored_tensor(layer["mask"]),
            }
            for layer in value["layers"]
        ]
        stored = {**value, "layers": layers, "_preview_cache": {}}
        self._entries.pop(key, None)
        self._entries[key] = stored
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def __delitem__(self, key):
        del self._entries[key]

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def clear(self):
        self._entries.clear()


def cached_layer_preview(staged, socket, cache_key, build_tensor, save_preview):
    cache = staged.setdefault("_preview_cache", {})
    cached = cache.get(socket)
    if cached is not None and cached["key"] == cache_key:
        return cached["preview"]
    preview = save_preview(build_tensor())
    if preview is not None:
        cache[socket] = {"key": cache_key, "preview": preview}
    return preview


def is_identity_projective_transform(corners, rotation):
    if float(rotation) != 0.0 or len(corners) != 4:
        return False
    return all(
        len(corner) == 2
        and float(corner[0]) == expected[0]
        and float(corner[1]) == expected[1]
        for corner, expected in zip(corners, _DEFAULT_CORNERS)
    )


def projective_warp(image, mask, corners, rotation=0.0):
    """Warp BHWC RGB and BHW alpha in pixel space through one inverse homography."""
    if is_identity_projective_transform(corners, rotation):
        return image, mask

    device, dtype = image.device, image.dtype
    height, width = image.shape[1:3]
    half_width = max(width - 1, 1) / 2
    half_height = max(height - 1, 1) / 2
    destination = torch.tensor(corners, device=device, dtype=dtype)
    destination = destination * destination.new_tensor((half_width, half_height))
    angle = torch.deg2rad(destination.new_tensor(float(rotation)))
    matrix = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle))),
            torch.stack((torch.sin(angle), torch.cos(angle))),
        )
    )
    destination = destination @ matrix.T
    minimum = destination.amin(dim=0)
    maximum = destination.amax(dim=0)
    output_width = max(1, int(torch.ceil(maximum[0] - minimum[0] - 1e-6).item()) + 1)
    output_height = max(1, int(torch.ceil(maximum[1] - minimum[1] - 1e-6).item()) + 1)
    destination = destination - minimum
    source = destination.new_tensor(
        (
            (0.0, 0.0),
            (float(width - 1), 0.0),
            (float(width - 1), float(height - 1)),
            (0.0, float(height - 1)),
        )
    )
    rows, values = [], []
    for (x, y), (u, v) in zip(source, destination):
        zero, one = x.new_tensor(0), x.new_tensor(1)
        rows.extend(
            [
                torch.stack((x, y, one, zero, zero, zero, -u * x, -u * y)),
                torch.stack((zero, zero, zero, x, y, one, -v * x, -v * y)),
            ]
        )
        values.extend((u, v))
    solved = torch.linalg.solve(torch.stack(rows), torch.stack(values))
    inverse = torch.linalg.inv(torch.cat((solved, solved.new_ones(1))).reshape(3, 3))
    ys = torch.arange(output_height, device=device, dtype=dtype)
    xs = torch.arange(output_width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    homogeneous = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
    mapped = homogeneous @ inverse.T
    mapped = mapped[..., :2] / mapped[..., 2:].clamp(min=1e-8)
    grid = torch.stack(
        (
            mapped[..., 0] * (2.0 / max(width - 1, 1)) - 1.0,
            mapped[..., 1] * (2.0 / max(height - 1, 1)) - 1.0,
        ),
        dim=-1,
    )
    rgba = torch.cat((image.movedim(-1, 1), mask.unsqueeze(1)), dim=1)
    warped = F.grid_sample(
        rgba,
        grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped[:, :3].movedim(1, -1), warped[:, 3]


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
        raise ValueError(
            "Layered foreground staging requires at least one foreground image."
        )
    layers = []
    full_alpha_by_socket = {}
    placements = _parse_layer_placements(placement_data)
    for key, foreground in foregrounds:
        if foreground.shape[-1] < 3:
            raise ValueError(
                f"Foreground input {key} must have at least three channels."
            )
        embedded_alpha = foreground[..., 3] if foreground.shape[-1] >= 4 else None
        foreground = foreground[..., :3]
        uses_embedded_alpha = embedded_alpha is not None
        if embedded_alpha is not None:
            # A supplied alpha channel is an explicit foreground matte. Preserve
            # it exactly instead of replacing it with a removal-model estimate.
            refined = (
                embedded_alpha[0]
                .to(device=foreground.device, dtype=foreground.dtype)
                .clamp(0.0, 1.0)
            )
            full_alpha = refined
        else:
            raw_mask = background_removal_model.encode_image(foreground)
            if not torch.is_tensor(raw_mask):
                raise ValueError(
                    f"Background removal returned an invalid mask for {key}."
                )
            if raw_mask.ndim == 4 and raw_mask.shape[1] == 1:
                raw_mask = raw_mask[:, 0]
            elif raw_mask.ndim == 4 and raw_mask.shape[-1] == 1:
                raw_mask = raw_mask[..., 0]
            if raw_mask.ndim != 3 or raw_mask.shape[0] != 1:
                raise ValueError(
                    f"Background removal must return one [batch, height, width] mask for {key}."
                )
            if raw_mask.shape[-2:] != foreground.shape[1:3]:
                raw_mask = _resize_composite_mask(
                    raw_mask,
                    foreground.shape[2],
                    foreground.shape[1],
                    mask_resize_method,
                )
            full_alpha = (
                raw_mask[0]
                .to(device=foreground.device, dtype=foreground.dtype)
                .clamp(0.0, 1.0)
            )
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
            raise ValueError(
                f"Background removal produced an empty foreground mask for {key}."
            )
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
        layers.append(
            {
                "socket": key,
                "image": cropped_image,
                "mask": cropped_mask,
                "flip_horizontal": flip_horizontal,
                "flip_vertical": flip_vertical,
                "uses_embedded_alpha": uses_embedded_alpha,
            }
        )
    staged = {"version": 1, "layers": layers}
    return (staged, full_alpha_by_socket) if retain_full_alpha else staged




def _apply_staged_layer_options(
    staged, foreground_blend, face_blend, face_feather_radius
):
    foreground_factor = 0.5 + 0.5 * float(foreground_blend)
    face_factor = 0.5 + 0.5 * float(face_blend)
    return {
        **staged,
        "layers": [
            {
                **layer,
                "blend_factor": face_factor
                if layer.get("is_face")
                else foreground_factor,
                **(
                    {"feather_radius": int(face_feather_radius)}
                    if layer.get("is_face")
                    else {}
                ),
            }
            for layer in staged["layers"]
        ],
    }


def _preview_staged_foregrounds(background, staged_foregrounds, feather_radius):
    if (
        not torch.is_tensor(background)
        or background.ndim != 4
        or background.shape[0] != 1
    ):
        raise ValueError(
            "Staged Layered Background Composite requires exactly one background image."
        )
    if background.shape[-1] < 3:
        raise ValueError("Background image must have at least three channels.")
    if (
        not isinstance(staged_foregrounds, dict)
        or staged_foregrounds.get("version") != 1
    ):
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
        editor_metadata["background_removal_model_name"] = staged_foregrounds[
            "background_removal_model_name"
        ]
    for layer in layers:
        crop = layer["image"]
        layer_feather = int(layer.get("feather_radius", feather_radius))
        applies_feather = bool(
            layer_feather
            and (layer.get("is_face") or not layer.get("uses_embedded_alpha", False))
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
                lambda image: _save_editor_preview(
                    image, f"UC_layered_{layer['socket']}"
                ),
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
    if (
        not torch.is_tensor(background)
        or background.ndim != 4
        or background.shape[0] != 1
    ):
        raise ValueError(
            "Staged Layered Background Composite requires exactly one background image."
        )
    if background.shape[-1] < 3:
        raise ValueError("Background image must have at least three channels.")
    if (
        not isinstance(staged_foregrounds, dict)
        or staged_foregrounds.get("version") != 1
    ):
        raise ValueError("Staged foreground data is missing or incompatible.")
    layers = staged_foregrounds.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("Staged foreground data contains no layers.")
    placements = _parse_layer_placements(placement_data)
    placement_version, _, _, workspace_padding = _parse_layer_payload(placement_data)
    paint = _parse_paint_layer(placement_data)
    layers_by_socket = {layer["socket"]: layer for layer in layers}
    if paint is not None:
        layers_by_socket[_PAINT_LAYER_KEY] = {
            "socket": _PAINT_LAYER_KEY,
            "is_paint": True,
            "paint": paint,
        }
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
        if layer.get("is_paint"):
            paint_rgba = load_staged_paint_rgba(
                layer["paint"],
                background_width,
                background_height,
                scene.device,
                scene.dtype,
            )
            paint_alpha = paint_rgba[0, ..., 3]
            included = layer["paint"]["included"]
            layer_mask = paint_alpha if included else torch.zeros_like(paint_alpha)
            layer_box = paint_alpha_bounds(layer_mask)
            if included:
                scene[0] = scene[0] * (1.0 - paint_alpha.unsqueeze(-1)) + paint_rgba[
                    0, ..., :3
                ] * paint_alpha.unsqueeze(-1)
                combined_mask[0] = combined_mask[0] + paint_alpha * (
                    1.0 - combined_mask[0]
                )
            layer_boxes.append(layer_box)
            layer_masks.append(layer_mask)
            continue
        crop = layer["image"]
        crop_mask = layer["mask"]
        preview_crop = crop
        preview_mask = crop_mask
        crop_height, crop_width = crop.shape[1:3]
        placement = placements.get(
            key,
            {
                **(
                    _DEFAULT_LAYER_PLACEMENT_V2
                    if placement_version >= 2
                    else _DEFAULT_LAYER_PLACEMENT
                ),
                "_version": placement_version,
            },
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
        target_longest = max(
            1, round(min(background_height, background_width) * placement["scale"])
        )
        scale = target_longest / max(crop_height, crop_width)
        placed_height = max(1, round(crop_height * scale))
        placed_width = max(1, round(crop_width * scale))
        resized_foreground = _resize_composite_image(
            crop, placed_width, placed_height, image_resize_method
        ).to(scene)
        resized_mask = _resize_composite_mask(
            crop_mask, placed_width, placed_height, mask_resize_method
        ).to(scene)
        if placement_version == 3:
            resized_foreground, resized_mask = projective_warp(
                resized_foreground,
                resized_mask,
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
            if (layer.get("uses_embedded_alpha", False) and not layer.get("is_face"))
            or not placed_feather
            else _feather_mask(resized_mask, -placed_feather)
        )
        offset_x, offset_y = _placement_offsets(
            background_width,
            background_height,
            placed_width,
            placed_height,
            placement,
            workspace_padding if placement_version >= 2 else 0.0,
        )
        slices = (
            None
            if excluded
            else _visible_placement_slices(
                background_width,
                background_height,
                placed_width,
                placed_height,
                offset_x,
                offset_y,
            )
        )
        layer_mask = scene.new_zeros((background_height, background_width))
        layer_box = {"x": 0, "y": 0, "width": 0, "height": 0}
        if slices is not None:
            (
                destination_top,
                destination_bottom,
                destination_left,
                destination_right,
                source_top,
                source_bottom,
                source_left,
                source_right,
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
            mask_region = combined_mask[
                0,
                destination_top:destination_bottom,
                destination_left:destination_right,
            ]
            blend_factor = float(layer.get("blend_factor", 1.0))
            placed_alpha = base_alpha * (1.0 - mask_region * (1.0 - blend_factor))
            placed_foreground = resized_foreground[
                0, source_top:source_bottom, source_left:source_right
            ]
            region = scene[
                0,
                destination_top:destination_bottom,
                destination_left:destination_right,
            ]
            scene[
                0,
                destination_top:destination_bottom,
                destination_left:destination_right,
            ] = region * (
                1.0 - placed_alpha.unsqueeze(-1)
            ) + placed_foreground * placed_alpha.unsqueeze(-1)
            combined_mask[
                0,
                destination_top:destination_bottom,
                destination_left:destination_right,
            ] = mask_region + base_alpha * (1.0 - mask_region)
        layer_boxes.append(layer_box)
        layer_masks.append(layer_mask)
        editor_layers.append(
            {
                "socket": key,
                "crop_width": crop_width,
                "crop_height": crop_height,
                "preview_crop": preview_crop,
                "preview_mask": preview_mask,
                "preview_feather": layer_feather,
                "preview_applies_feather": bool(
                    layer_feather
                    and (
                        layer.get("is_face")
                        or not layer.get("uses_embedded_alpha", False)
                    )
                ),
                "flip_horizontal": staged_flip,
                "flip_vertical": staged_flip_vertical,
                "is_face": bool(layer.get("is_face", False)),
                "included": not excluded,
                "blend_factor": float(layer.get("blend_factor", 1.0)),
            }
        )

    editor_metadata = {
        "version": 1,
        "background": {"width": background_width, "height": background_height},
        "layers": [],
    }
    if staged_foregrounds.get("background_removal_model_name"):
        editor_metadata["background_removal_model_name"] = staged_foregrounds[
            "background_removal_model_name"
        ]
    if stage_mode:
        editor_metadata["stage_mode"] = stage_mode
    for layer in editor_layers:
        entry = {
            key: layer[key]
            for key in (
                "socket",
                "crop_width",
                "crop_height",
                "flip_horizontal",
                "flip_vertical",
                "is_face",
                "included",
                "blend_factor",
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
                (
                    "rgba-v1",
                    layer["preview_feather"] if layer["preview_applies_feather"] else 0,
                ),
                build_preview_tensor,
                lambda image: _save_editor_preview(
                    image, f"UC_layered_{layer['socket']}"
                ),
            )
        except Exception:
            logging.warning(
                "Unable to create staged editor cutout preview for %s.",
                layer["socket"],
                exc_info=True,
            )
        editor_metadata["layers"].append(entry)
    ordered_masks = (
        torch.stack(layer_masks)
        if layer_masks
        else scene.new_zeros((0, background_height, background_width))
    )
    return io.NodeOutput(
        scene,
        combined_mask,
        [layer_boxes] if layer_boxes else [],
        ordered_masks,
        ui={"uc_layered_scene_editor": [editor_metadata]},
    )
