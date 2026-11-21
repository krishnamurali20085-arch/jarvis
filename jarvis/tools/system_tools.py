"""Narrowly scoped, OS-aware desktop and system operations."""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from datetime import datetime
from typing import Any

import psutil
from PIL import ImageGrab

logger = logging.getLogger(__name__)

_APPLICATIONS: dict[str, dict[str, list[str]]] = {
    "windows": {
        "calculator": ["calc.exe"],
        "text_editor": ["notepad.exe"],
        "paint": ["mspaint.exe"],
    },
    "darwin": {
        "calculator": ["open", "-a", "Calculator"],
        "text_editor": ["open", "-a", "TextEdit"],
    },
    "linux": {
        "calculator": ["gnome-calculator"],
        "text_editor": ["gedit"],
    },
}


def _platform_key() -> str:
    system_name = platform.system().lower()
    if system_name == "windows":
        return "windows"
    if system_name == "darwin":
        return "darwin"
    if system_name == "linux":
        return "linux"
    return system_name


def get_system_status() -> dict[str, Any]:
    """Return CPU, memory, and available battery charge information."""
    status: dict[str, Any] = {"platform": platform.platform()}
    try:
        status["cpu_percent"] = psutil.cpu_percent(interval=None)
    except (OSError, psutil.Error) as exc:
        status["cpu_error"] = str(exc)
    try:
        memory = psutil.virtual_memory()
        status["memory"] = {
            "total_bytes": memory.total,
            "available_bytes": memory.available,
            "percent_used": memory.percent,
        }
    except (OSError, psutil.Error) as exc:
        status["memory_error"] = str(exc)
    try:
        battery = psutil.sensors_battery()
        status["battery"] = (
            {
                "charge_percent": battery.percent,
                "plugged_in": battery.power_plugged,
                "health": None,
                "health_note": (
                    "Battery charge is available; hardware health is not exposed "
                    "by the portable system metrics API."
                ),
            }
            if battery is not None
            else None
        )
    except (AttributeError, OSError, psutil.Error) as exc:
        status["battery_error"] = str(exc)
    return {"ok": True, "data": status}


def open_application(application: str) -> dict[str, Any]:
    """Launch a small, fixed set of built-in applications."""
    alias = application.strip().lower()
    platform_key = _platform_key()
    command = _APPLICATIONS.get(platform_key, {}).get(alias)
    if command is None:
        allowed = sorted(_APPLICATIONS.get(platform_key, {}))
        return {
            "ok": False,
            "error": {
                "code": "unsupported_application",
                "message": f"Application is not allowlisted. Available: {allowed}",
            },
        }

    if platform_key == "linux":
        executable = shutil.which(command[0])
        if executable is None:
            return {
                "ok": False,
                "error": {
                    "code": "application_not_found",
                    "message": f"{command[0]} is not installed or not on PATH.",
                },
            }
        command = [executable, *command[1:]]

    try:
        process = subprocess.Popen(
            command,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=platform_key != "windows",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Could not launch allowlisted application %s: %s", alias, exc)
        return {
            "ok": False,
            "error": {
                "code": "launch_failed",
                "message": f"Could not launch {alias}: {exc}",
            },
        }
    return {"ok": True, "data": {"application": alias, "process_id": process.pid}}


def close_application(application: str) -> dict[str, Any]:
    """Gracefully close matching processes from the fixed application allowlist."""
    alias = application.strip().lower()
    process_names = {
        "calculator": {"calculator.exe", "calculator", "gnome-calculator"},
        "text_editor": {"notepad.exe", "textedit", "gedit"},
        "paint": {"mspaint.exe"},
    }
    names = process_names.get(alias)
    if names is None:
        return {
            "ok": False,
            "error": {
                "code": "unsupported_application",
                "message": "Only allowlisted application names can be closed.",
            },
        }

    matches: list[psutil.Process] = []
    try:
        for process in psutil.process_iter(["name"]):
            try:
                process_name = process.info.get("name")
                if process_name and process_name.casefold() in names:
                    matches.append(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not matches:
            return {
                "ok": True,
                "data": {"application": alias, "closed_processes": 0},
            }

        for process in matches:
            try:
                process.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                logger.info(
                    "Could not request graceful close for PID %s: %s", process.pid, exc
                )
        _, alive = psutil.wait_procs(matches, timeout=3)
        result: dict[str, Any] = {
            "ok": not alive,
            "data": {
                "application": alias,
                "closed_processes": len(matches) - len(alive),
                "still_running_process_ids": [process.pid for process in alive],
            },
        }
        if alive:
            result["error"] = {
                "code": "process_still_running",
                "message": "Some processes did not close; they were not force-killed.",
            }
        return result
    except (OSError, psutil.Error) as exc:
        logger.warning("Could not close allowlisted application %s: %s", alias, exc)
        return {
            "ok": False,
            "error": {"code": "close_failed", "message": str(exc)},
        }


def set_system_volume(level: int) -> dict[str, Any]:
    """Set output volume from 0 to 100 using a platform-specific native tool."""
    if isinstance(level, bool) or not 0 <= level <= 100:
        raise ValueError("Volume level must be an integer between 0 and 100.")
    platform_key = _platform_key()
    try:
        if platform_key == "windows":
            from ctypes import POINTER, cast

            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            device = AudioUtilities.GetSpeakers()
            interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            volume.SetMasterVolumeLevelScalar(level / 100, None)
        elif platform_key == "darwin":
            subprocess.run(
                ["osascript", "-e", f"set volume output volume {level}"],
                check=True,
                shell=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        elif platform_key == "linux":
            executable = shutil.which("pactl")
            if executable is None:
                return {
                    "ok": False,
                    "error": {
                        "code": "unsupported",
                        "message": "Volume control requires PulseAudio/PipeWire `pactl`.",
                    },
                }
            subprocess.run(
                [executable, "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"],
                check=True,
                shell=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        else:
            return {
                "ok": False,
                "error": {
                    "code": "unsupported",
                    "message": f"Volume control is not implemented for {platform_key}.",
                },
            }
    except ImportError as exc:
        return {
            "ok": False,
            "error": {
                "code": "optional_dependency_missing",
                "message": f"Install the platform audio dependency to control volume: {exc}",
            },
        }
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.warning("Volume adjustment failed: %s", exc)
        return {
            "ok": False,
            "error": {"code": "volume_control_failed", "message": str(exc)},
        }
    return {"ok": True, "data": {"volume_percent": level}}


def set_screen_brightness(level: int) -> dict[str, Any]:
    """Set screen brightness from 0 to 100 where the platform supports it."""
    if isinstance(level, bool) or not 0 <= level <= 100:
        raise ValueError("Brightness level must be an integer between 0 and 100.")
    platform_key = _platform_key()
    if platform_key == "darwin":
        return {
            "ok": False,
            "error": {
                "code": "unsupported",
                "message": "Built-in macOS brightness control is not available in this tool.",
            },
        }
    try:
        if platform_key == "windows":
            import screen_brightness_control

            screen_brightness_control.set_brightness(level)
        elif platform_key == "linux":
            executable = shutil.which("brightnessctl")
            if executable is None:
                return {
                    "ok": False,
                    "error": {
                        "code": "unsupported",
                        "message": "Brightness control requires `brightnessctl`.",
                    },
                }
            subprocess.run(
                [executable, "set", f"{level}%"],
                check=True,
                shell=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        else:
            return {
                "ok": False,
                "error": {
                    "code": "unsupported",
                    "message": f"Brightness control is not implemented for {platform_key}.",
                },
            }
    except ImportError as exc:
        return {
            "ok": False,
            "error": {
                "code": "optional_dependency_missing",
                "message": f"Install screen-brightness-control to change brightness: {exc}",
            },
        }
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.warning("Brightness adjustment failed: %s", exc)
        return {
            "ok": False,
            "error": {"code": "brightness_control_failed", "message": str(exc)},
        }
    return {"ok": True, "data": {"brightness_percent": level}}


def take_screenshot() -> dict[str, Any]:
    """Capture the current desktop to the configured workspace."""
    from jarvis.tools.file_tools import resolve_workspace_path

    screenshot_dir = resolve_workspace_path("screenshots")
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir = resolve_workspace_path("screenshots")
    filename = f"screenshot-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.png"
    destination = screenshot_dir / filename
    try:
        image = ImageGrab.grab()
        image.save(destination, format="PNG")
    except (OSError, ValueError) as exc:
        logger.warning("Could not capture screenshot: %s", exc)
        return {
            "ok": False,
            "error": {
                "code": "screenshot_failed",
                "message": (
                    "Could not capture the desktop. Check screen-capture permissions "
                    f"and desktop-session support: {exc}"
                ),
            },
        }
    return {"ok": True, "data": {"path": str(destination)}}
