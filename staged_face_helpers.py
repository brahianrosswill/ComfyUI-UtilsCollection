import logging
import os

import numpy as np
import torch

from .model_assets import ensure_huggingface_model


_FACE_MODEL_CACHE = {"path": None, "model": None}


def load_face_model():
    import comfy.utils
    from comfy_extras.nodes_mediapipe import FaceLandmarkerModel

    filename = "mediapipe_face_fp32.safetensors"
    path = ensure_huggingface_model(
        "detection",
        filename,
        "Comfy-Org/mediapipe",
        f"detection/{filename}",
    )
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


def _run_detection_batch(model, images_uint8, maximum_faces, threshold):
    return model.detect_batch(
        images_uint8, num_faces=max(1, int(maximum_faces)) * 3,
        score_thresh=threshold, variant="full",
    )


def _merged_faces(candidates, maximum_faces):
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


def detect_faces_adaptive(model, image_uint8, maximum_faces, threshold):
    results, failed = detect_many_or_warn(
        model, [("", image_uint8)], maximum_faces, threshold
    )
    if failed:
        raise RuntimeError("MediaPipe face detection failed.")
    return results[""]


def detect_many_or_warn(model, images, maximum_faces, threshold):
    threshold = min(max(float(threshold), 0.01), 1.0)
    thresholds = []
    for value in (threshold, max(0.05, round(threshold * 0.7, 4)), max(0.05, round(threshold * 0.5, 4))):
        if value not in thresholds:
            thresholds.append(value)

    results = {socket: [] for socket, _ in images}
    pending = list(images)
    failed = set()
    for pass_threshold in thresholds:
        if not pending:
            break
        try:
            batches = _run_detection_batch(
                model, [image for _, image in pending], maximum_faces, pass_threshold
            )
            pass_results = list(zip(pending, batches))
        except Exception:
            pass_results = []
            for item in pending:
                try:
                    batches = _run_detection_batch(
                        model, [item[1]], maximum_faces, pass_threshold
                    )
                    pass_results.append((item, batches[0]))
                except Exception:
                    failed.add(item[0])
                    logging.warning(
                        "MediaPipe face detection failed for %s; retaining ordinary foreground only.",
                        item[0], exc_info=True,
                    )

        next_pending = []
        for (socket, image), candidates in pass_results:
            if socket in failed:
                continue
            if candidates:
                results[socket] = _merged_faces(candidates, maximum_faces)
            else:
                next_pending.append((socket, image))
        pending = next_pending
    return results, failed


def detect_or_warn(model, image_uint8, maximum_faces, threshold, socket):
    results, failed = detect_many_or_warn(
        model, [(socket, image_uint8)], maximum_faces, threshold
    )
    return results[socket], socket in failed
