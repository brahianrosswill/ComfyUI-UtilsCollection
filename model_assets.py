import os
import tempfile
import threading
from urllib.parse import quote


_DOWNLOAD_LOCKS = {}
_DOWNLOAD_LOCKS_GUARD = threading.Lock()


def _asset_lock(key):
    with _DOWNLOAD_LOCKS_GUARD:
        return _DOWNLOAD_LOCKS.setdefault(key, threading.Lock())


def ensure_huggingface_model(category, filename, repo_id, repo_path):
    """Return an existing model path or download it into a ComfyUI model folder."""
    import folder_paths

    try:
        existing = folder_paths.get_full_path_or_raise(category, filename)
    except Exception:
        existing = None
    if existing and os.path.isfile(existing) and os.path.getsize(existing) > 0:
        return existing

    directories = list(folder_paths.get_folder_paths(category))
    if not directories:
        raise ValueError(f"ComfyUI has no registered model directory for {category!r}.")
    destination_directory = os.path.abspath(directories[0])
    destination = os.path.join(destination_directory, filename)
    key = (category, os.path.normcase(destination))

    with _asset_lock(key):
        if os.path.isfile(destination) and os.path.getsize(destination) > 0:
            return destination
        url = f"https://huggingface.co/{repo_id}/resolve/main/{quote(repo_path, safe='/')}"
        temporary_path = None
        try:
            import requests

            os.makedirs(destination_directory, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=f".{filename}.", suffix=".part",
                dir=destination_directory, delete=False,
            ) as temporary:
                temporary_path = temporary.name
                with requests.get(
                    url, stream=True, allow_redirects=True, timeout=(10, 300)
                ) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            temporary.write(chunk)
            if os.path.getsize(temporary_path) <= 0:
                raise ValueError(f"Downloaded {repo_path} is empty.")
            os.replace(temporary_path, destination)
            temporary_path = None
        except Exception as exc:
            raise ValueError(
                f"Unable to download {url} to {destination}."
            ) from exc
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)
        return destination
