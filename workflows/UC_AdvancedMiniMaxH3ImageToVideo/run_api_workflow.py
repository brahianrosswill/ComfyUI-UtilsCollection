"""Upload the example references and queue its ComfyUI API workflow."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import uuid
from pathlib import Path
from urllib import error, request


WORKFLOW_FILE = Path(__file__).with_name("QwenOnly_8Image_1024VLM_API.json")
IMAGE_NAMES = (
    "ComfyUI_withoutcat_ref_00.00s_00001_.png",
    "ComfyUI_withoutcat_ref_01.21s_00001_.png",
    "ComfyUI_withoutcat_ref_02.46s_00001_.png",
    "ComfyUI_withoutcat_ref_05.30s_00001_.png",
    "ComfyUI_withoutcat_ref_06.55s_00001_.png",
    "ComfyUI_withoutcat_ref_07.80s_00001_.png",
    "ComfyUI_withoutcat_ref_10.84s_00001_.png",
    "ComfyUI_withoutcat_ref_12.10s_00001_.png",
)


def _json_request(url: str, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with request.urlopen(req) as response:
        return json.load(response)


def _upload_image(server: str, path: Path) -> str:
    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body = b"".join(
        (
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="image"; '
                f'filename="{path.name}"\r\n'
            ).encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            path.read_bytes(),
            b"\r\n",
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n',
            f"--{boundary}--\r\n".encode(),
        )
    )
    req = request.Request(f"{server}/upload/image", data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with request.urlopen(req) as response:
        result = json.load(response)
    subfolder = result.get("subfolder", "")
    return f"{subfolder}/{result['name']}" if subfolder else result["name"]


def _wait_for_history(server: str, prompt_id: str, poll_seconds: float):
    while True:
        history = _json_request(f"{server}/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    server = args.server.rstrip("/")

    paths = [args.reference_dir / name for name in IMAGE_NAMES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error("missing reference images: " + ", ".join(missing))

    workflow = json.loads(WORKFLOW_FILE.read_text(encoding="utf-8"))
    load_nodes = sorted(
        (key for key, node in workflow.items() if node["class_type"] == "LoadImage"),
        key=int,
    )
    if len(load_nodes) != len(paths):
        raise RuntimeError("API workflow does not contain the expected eight LoadImage nodes.")

    for node_id, path in zip(load_nodes, paths):
        workflow[node_id]["inputs"]["image"] = _upload_image(server, path)
    if args.seed is not None:
        noise_nodes = [
            node for node in workflow.values() if node["class_type"] == "RandomNoise"
        ]
        if len(noise_nodes) != 1:
            raise RuntimeError("API workflow does not contain exactly one RandomNoise node.")
        noise_nodes[0]["inputs"]["noise_seed"] = args.seed

    queued = _json_request(f"{server}/prompt", {"prompt": workflow})
    prompt_id = queued["prompt_id"]
    sys.stdout.write(f"Queued prompt {prompt_id}\n")
    history = _wait_for_history(server, prompt_id, args.poll_seconds)
    status = history.get("status", {})
    if status.get("status_str") == "error":
        sys.stderr.write(json.dumps(status, indent=2) + "\n")
        return 1
    sys.stdout.write(json.dumps(history.get("outputs", {}), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except error.URLError as exc:
        raise SystemExit(f"ComfyUI API request failed: {exc}") from exc
