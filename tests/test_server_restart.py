import asyncio
import os
import pathlib
import sys
import types
from types import SimpleNamespace


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_restart_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_restart_test import server_restart


def test_restart_argv_preserves_regular_launch_arguments(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "executable", "/venv/bin/python")

    assert server_restart._restart_argv(["main.py", "--listen", "0.0.0.0"]) == [
        "/venv/bin/python",
        "main.py",
        "--listen",
        "0.0.0.0",
    ]


def test_restart_argv_preserves_module_launch(monkeypatch):
    monkeypatch.setattr(sys, "executable", "/venv/bin/python")

    assert server_restart._restart_argv(
        [os.path.join("ComfyUI", "__main__.py"), "--listen"]
    ) == ["/venv/bin/python", "-m", "ComfyUI", "--listen"]


def test_restart_argv_matches_windows_legacy_launch(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "executable", "C:\\ComfyUI\\python.exe")

    assert server_restart._restart_argv(
        ["main.py", "--windows-standalone-build", "--listen"]
    ) == ['"C:\\ComfyUI\\python.exe"', '"main.py"', "--listen"]


def test_restart_handler_rejects_simple_form_requests(monkeypatch):
    called = False

    def restart():
        nonlocal called
        called = True

    monkeypatch.setattr(server_restart, "_restart_comfyui", restart)
    response = asyncio.run(
        server_restart._handle_restart(
            SimpleNamespace(content_type="application/x-www-form-urlencoded")
        )
    )

    assert response.status == 415
    assert not called


def test_restart_handler_accepts_json(monkeypatch):
    called = False

    def restart():
        nonlocal called
        called = True

    monkeypatch.setattr(server_restart, "_restart_comfyui", restart)
    response = asyncio.run(
        server_restart._handle_restart(SimpleNamespace(content_type="application/json"))
    )

    assert response.status == 200
    assert called
