import os
from urllib.parse import quote


def require_huggingface_model(category, filename, repo_id, repo_path):
    """Return an existing model path or raise an actionable installation error."""
    import folder_paths

    try:
        existing = folder_paths.get_full_path_or_raise(category, filename)
    except Exception:
        existing = None
    if existing and os.path.isfile(existing) and os.path.getsize(existing) > 0:
        return existing

    directories = list(folder_paths.get_folder_paths(category))
    url = f"https://huggingface.co/{repo_id}/blob/main/{quote(repo_path, safe='/')}"
    if not directories:
        raise ValueError(
            f"Required model {filename} was not found, and ComfyUI has no registered "
            f"model directory for {category!r}. Download it from {url}."
        )
    expected = [os.path.abspath(os.path.join(directory, filename)) for directory in directories]
    locations = "\n".join(f"  - {path}" for path in expected)
    raise ValueError(
        f"Required model {filename} was not found. Download it from:\n"
        f"  {url}\n"
        f"Place it in one of ComfyUI's registered {category} directories:\n{locations}"
    )
