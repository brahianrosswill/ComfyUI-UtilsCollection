import os
import sys

from aiohttp import web
from server import PromptServer


RESTART_ROUTE = "/utils_collection/restart"
_route_registered = False


def _restart_argv(argv=None):
    sys_argv = list(sys.argv if argv is None else argv)
    if "--windows-standalone-build" in sys_argv:
        sys_argv.remove("--windows-standalone-build")

    if sys_argv[0].endswith("__main__.py"):
        module_name = os.path.basename(os.path.dirname(sys_argv[0]))
        return [sys.executable, "-m", module_name, *sys_argv[1:]]
    if sys.platform.startswith("win32"):
        return [f'"{sys.executable}"', f'"{sys_argv[0]}"', *sys_argv[1:]]
    return [sys.executable, *sys_argv]


def _restart_comfyui():
    try:
        sys.stdout.close_log()
    except Exception:
        pass

    cli_session = os.environ.get("__COMFY_CLI_SESSION__")
    if cli_session:
        with open(f"{cli_session}.reboot", "w"):
            pass
        print("\nRestarting...\n", flush=True)
        raise SystemExit(0)

    restart_argv = _restart_argv()
    print(f"\nRestarting...\nCommand: {restart_argv}", flush=True)
    os.execv(sys.executable, restart_argv)


async def _handle_restart(request):
    if request.content_type != "application/json":
        return web.json_response(
            {"error": "Restart requires an application/json request."},
            status=415,
        )

    _restart_comfyui()
    return web.json_response({"status": "restarting"})


def register_restart_route():
    global _route_registered
    if _route_registered:
        return
    PromptServer.instance.routes.post(RESTART_ROUTE)(_handle_restart)
    _route_registered = True
