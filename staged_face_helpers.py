import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F


_FACE_MODEL_CACHE = {"path": None, "model": None}


def load_face_model():
    import comfy.utils
    import folder_paths
    from comfy_extras.nodes_mediapipe import FaceLandmarkerModel

    filename = "mediapipe_face_fp32.safetensors"
    try:
        path = folder_paths.get_full_path_or_raise("detection", filename)
    except Exception as exc:
        raise ValueError(
            f"The MediaPipe face model is missing. Install it as "
            f"{os.path.join('models', 'detection', filename)}."
        ) from exc
    path = os.path.normcase(os.path.abspath(path))
    if _FACE_MODEL_CACHE["path"] == path:
        return _FACE_MODEL_CACHE["model"]
    try:
        model = FaceLandmarkerModel(comfy.utils.load_torch_file(path, safe_load=True))
    except Exception as exc:
        raise ValueError(f"Comfy Core could not load {filename} as a face detection model.") from exc
    _FACE_MODEL_CACHE.update(path=path, model=model)
    return model


def _box_iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-6)


def _same_face(a, b):
    if _box_iou(a["bbox_xyxy"], b["bbox_xyxy"]) >= 0.3:
        return True
    boxes = [np.asarray(face["bbox_xyxy"], dtype=np.float32) for face in (a, b)]
    centers = [(box[:2] + box[2:]) * 0.5 for box in boxes]
    scales = [max(box[2] - box[0], box[3] - box[1], 1.0) for box in boxes]
    return np.linalg.norm(centers[0] - centers[1]) <= 0.35 * max(scales) and (
        min(scales) / max(scales) >= 0.45
    )


def _merge_cluster(cluster):
    weights = np.asarray([
        max(float(face.get("score", 0.0)), 1e-4)
        * max(float(face.get("presence", 1.0)), 1e-4)
        for face in cluster
    ], dtype=np.float32)
    weights /= weights.sum()
    boxes = np.stack([np.asarray(face["bbox_xyxy"], dtype=np.float32) for face in cluster])
    # Use the union for extraction so a partial high-confidence observation cannot
    # cut away regions supported by another threshold pass.
    merged = dict(max(cluster, key=lambda face: float(face.get("score", 0.0))))
    merged["bbox_xyxy"] = np.array([
        boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()
    ], dtype=np.float32)
    merged["center_xy"] = np.sum(
        ((boxes[:, :2] + boxes[:, 2:]) * 0.5) * weights[:, None], axis=0
    )
    merged["score"] = float(max(float(face.get("score", 0.0)) for face in cluster))
    merged["observations"] = len(cluster)
    return merged


def _run_detection(model, image_uint8, maximum_faces, threshold):
    return model.detect_batch(
        [image_uint8], num_faces=max(1, int(maximum_faces)) * 3,
        score_thresh=threshold, variant="full",
    )[0] or []


def detect_faces_adaptive(model, image_uint8, maximum_faces, threshold):
    threshold = min(max(float(threshold), 0.01), 1.0)
    candidates = _run_detection(model, image_uint8, maximum_faces, threshold)
    # Lower confidence only as a recovery path. Running every lower threshold
    # unconditionally turns detector noise into dozens of placeable layers.
    if not candidates:
        candidates = _run_detection(
            model, image_uint8, maximum_faces, max(0.05, round(threshold * 0.7, 4))
        )
    if not candidates:
        candidates = _run_detection(
            model, image_uint8, maximum_faces, max(0.05, round(threshold * 0.5, 4))
        )

    clusters = []
    for face in sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        matching = next((cluster for cluster in clusters if any(_same_face(face, other) for other in cluster)), None)
        if matching is None:
            clusters.append([face])
        else:
            matching.append(face)
    merged = [_merge_cluster(cluster) for cluster in clusters]
    merged.sort(key=lambda face: (float(face["center_xy"][1]), float(face["center_xy"][0])))
    return merged[:max(1, int(maximum_faces))]


def detect_or_warn(model, image_uint8, maximum_faces, threshold, socket):
    try:
        return detect_faces_adaptive(model, image_uint8, maximum_faces, threshold), False
    except Exception:
        logging.warning(
            "MediaPipe face detection failed for %s; retaining ordinary foreground only.",
            socket, exc_info=True,
        )
        return [], True


def projective_warp(image, mask, corners, rotation=0.0):
    """Warp BHWC RGB and BHW alpha through the same inverse homography."""
    device, dtype = image.device, image.dtype
    destination = torch.tensor(corners, device=device, dtype=dtype)
    angle = math.radians(float(rotation))
    matrix = destination.new_tensor([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    destination = destination @ matrix.T
    source = destination.new_tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    rows, values = [], []
    for (x, y), (u, v) in zip(source, destination):
        zero, one = x.new_tensor(0), x.new_tensor(1)
        rows.extend([
            torch.stack((x, y, one, zero, zero, zero, -u*x, -u*y)),
            torch.stack((zero, zero, zero, x, y, one, -v*x, -v*y)),
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
    warped_image = F.grid_sample(
        image.movedim(-1, 1), grid.unsqueeze(0), mode="bilinear",
        padding_mode="zeros", align_corners=True,
    ).movedim(1, -1)
    warped_mask = F.grid_sample(
        mask.unsqueeze(1), grid.unsqueeze(0), mode="bilinear",
        padding_mode="zeros", align_corners=True,
    ).squeeze(1)
    return warped_image, warped_mask
