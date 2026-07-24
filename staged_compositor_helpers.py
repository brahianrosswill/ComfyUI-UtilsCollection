from collections import OrderedDict
from collections.abc import MutableMapping

import torch
import torch.nn.functional as F


_DEFAULT_CORNERS = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))


def _stored_tensor(tensor):
    import comfy.model_management

    stored = tensor.detach().to(
        device=comfy.model_management.intermediate_device(),
        dtype=comfy.model_management.intermediate_dtype(),
    ).contiguous()
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
    """Warp BHWC RGB and BHW alpha through one shared inverse homography."""
    if is_identity_projective_transform(corners, rotation):
        return image, mask

    device, dtype = image.device, image.dtype
    destination = torch.tensor(corners, device=device, dtype=dtype)
    angle = torch.deg2rad(destination.new_tensor(float(rotation)))
    matrix = torch.stack((
        torch.stack((torch.cos(angle), -torch.sin(angle))),
        torch.stack((torch.sin(angle), torch.cos(angle))),
    ))
    destination = destination @ matrix.T
    source = destination.new_tensor(_DEFAULT_CORNERS)
    rows, values = [], []
    for (x, y), (u, v) in zip(source, destination):
        zero, one = x.new_tensor(0), x.new_tensor(1)
        rows.extend([
            torch.stack((x, y, one, zero, zero, zero, -u * x, -u * y)),
            torch.stack((zero, zero, zero, x, y, one, -v * x, -v * y)),
        ])
        values.extend((u, v))
    solved = torch.linalg.solve(torch.stack(rows), torch.stack(values))
    inverse = torch.linalg.inv(torch.cat((solved, solved.new_ones(1))).reshape(3, 3))
    height, width = image.shape[1:3]
    ys = torch.linspace(-1, 1, height, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    homogeneous = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
    mapped = homogeneous @ inverse.T
    grid = mapped[..., :2] / mapped[..., 2:].clamp(min=1e-8)
    rgba = torch.cat((image.movedim(-1, 1), mask.unsqueeze(1)), dim=1)
    warped = F.grid_sample(
        rgba, grid.unsqueeze(0), mode="bilinear",
        padding_mode="zeros", align_corners=True,
    )
    return warped[:, :3].movedim(1, -1), warped[:, 3]
