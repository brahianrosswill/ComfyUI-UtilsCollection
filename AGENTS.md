# Agent Instructions

## Test environment

- Never run this repository's Python tests with an unverified `python` or `pytest` command, and never silently fall back to the system Python installation.
- Before running tests, determine which Python virtual environment the user intends this ComfyUI installation to use. Prefer ComfyUI's parent `.venv` or `venv` only when it exists and is clearly associated with the containing ComfyUI checkout.
- If no associated environment exists, more than one plausible environment exists, or the intended environment cannot be established from the current activated environment or repository context, ask the user which environment to use before running tests.
- Do not assume a fixed filesystem location, operating system, environment directory name, or Python executable path.
- When using a parent ComfyUI environment, run pytest from that ComfyUI root and pass `--import-mode=importlib`; otherwise pytest may import this custom node's `__init__.py` without a package context.
- Use a repository-local `--basetemp` directory because the global Windows pytest temp directory may be inaccessible. Remove that temporary directory after the run.
- For example, when the containing ComfyUI checkout has a Windows `.venv`, the focused visual-fusion suite can be run from the ComfyUI root with:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -q --import-mode=importlib --basetemp custom_nodes\ComfyUI-UtilsCollection\.pytest-tmp custom_nodes/ComfyUI-UtilsCollection/tests/test_visual_fusion.py custom_nodes/ComfyUI-UtilsCollection/tests/test_encoder_correctness.py
  ```

- Do not run model inference unless the user explicitly requests it.
