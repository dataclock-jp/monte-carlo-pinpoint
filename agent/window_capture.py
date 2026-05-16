"""
window_capture.py

W4 v0.1 — per-window capture infrastructure for V2A (decision 058).

Two public helpers plus matching MCP tools in ``mcp_server.py``:

    list_windows_structured(include_hidden=False, include_minimized=True)
        → list[dict]   structured top-level window metadata

    capture_window(title_regex="", handle=0, occluded=True, resize_width=0,
                   save_path="")
        → PIL.Image    single-window bitmap, occluded content preserved

Why this exists alongside the pre-existing ``list_windows()`` and
``screenshot(window=...)``:

* ``list_windows()`` returns a newline-joined human-readable string; it
  has no handles, bounds, visibility flags, or app-name. Automation
  cannot look up a specific window by regex and reuse its handle across
  calls.
* ``screenshot(window=keyword)`` captures the *screen region* the window
  occupies. If the window is partially or fully occluded by another
  window (common during multi-agent supervision), the region contains
  whatever is on top, not the target window's own pixels.

``capture_window_printwindow`` closes both gaps by calling the Win32
``PrintWindow`` API with ``PW_RENDERFULLCONTENT`` (0x02), which asks the
window to render its own content even when occluded. The region-capture
path is kept as an automatic fallback for windows that don't respond
to PrintWindow (some DirectX / OpenGL surfaces).

Windows-only in v0.1. Non-Win32 callers must check ``IS_WINDOWS`` before
calling any helper. Cross-platform is deferred to v0.2 / v0.3.
"""
from __future__ import annotations

import re as _re
import sys
from typing import Any, Dict, List, Optional

from PIL import Image as _Image


IS_WINDOWS = sys.platform == "win32"


class WindowCaptureError(RuntimeError):
    """Raised for all window-capture failures (bad handle, PrintWindow
    refusal, no matching title, non-Win32 platform, etc.)."""


# Window titles we never enumerate. Matches the existing `list_windows`
# exclusion set in `mcp_server.py` plus a few additional system surfaces
# that tend to be invisible but enumerated by EnumWindows.
_SYSTEM_WINDOW_TITLES = {
    "",
    "program manager",
    "settings",
    "microsoft text input application",
    "windows input experience",
    "msctfime ui",
    "default ime",
    "nvidia geforce overlay",  # driver overlay, not user-facing
}
_MIN_TITLE_LEN = 3


def _require_windows() -> None:
    if not IS_WINDOWS:
        raise WindowCaptureError(
            "window_capture helpers are Windows-only in v0.1; non-Win32 "
            "platforms return early from the calling MCP tool instead of "
            "reaching this helper"
        )


def _exe_basename_for_hwnd(hwnd: int) -> str:
    """Best-effort process name lookup; silently returns '' on failure."""
    try:
        import ctypes
        import ctypes.wintypes
        import os as _os
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        pid = ctypes.wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return ""
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h_process = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if not h_process:
            return ""
        try:
            buf_len = ctypes.wintypes.DWORD(260)
            buf = ctypes.create_unicode_buffer(buf_len.value)
            if kernel32.QueryFullProcessImageNameW(
                    h_process, 0, buf, ctypes.byref(buf_len)):
                return _os.path.basename(buf.value)
        finally:
            kernel32.CloseHandle(h_process)
    except Exception:
        return ""
    return ""


def list_windows_structured(
    include_hidden: bool = False,
    include_minimized: bool = True,
) -> List[Dict[str, Any]]:
    """Enumerate top-level windows with structured metadata.

    Args:
        include_hidden: include windows for which ``IsWindowVisible`` is
            false. Default False: hidden windows are noisy (tooltips,
            work areas) and rarely the automation target.
        include_minimized: include windows for which ``IsIconic`` is
            true. Default True: a minimized window is still a legitimate
            target for PrintWindow capture.

    Returns:
        List of dicts with keys:

            handle         : int   HWND, session-unique
            title          : str   GetWindowTextW
            app_name       : str   exe basename, or '' on failure
            visible        : bool
            minimized      : bool
            bounds         : dict  {"x", "y", "w", "h"} from GetWindowRect
            is_foreground  : bool  GetForegroundWindow match

    System surface titles (empty, "Program Manager", IME services, etc.)
    are filtered out even when ``include_hidden`` is true.
    """
    _require_windows()

    import ctypes
    import ctypes.wintypes
    user32 = ctypes.windll.user32
    fg_hwnd = user32.GetForegroundWindow()

    results: List[Dict[str, Any]] = []

    WNDENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)

    def _callback(hwnd, _lparam):
        try:
            visible = bool(user32.IsWindowVisible(hwnd))
            if not visible and not include_hidden:
                return True

            minimized = bool(user32.IsIconic(hwnd))
            if minimized and not include_minimized:
                return True

            length = user32.GetWindowTextLengthW(hwnd)
            title = ""
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value.strip()

            title_low = title.lower()
            if (not title
                    or title_low in _SYSTEM_WINDOW_TITLES
                    or len(title) < _MIN_TITLE_LEN):
                return True

            rect = ctypes.wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return True

            results.append({
                "handle": int(hwnd),
                "title": title,
                "app_name": _exe_basename_for_hwnd(hwnd),
                "visible": visible,
                "minimized": minimized,
                "bounds": {
                    "x": int(rect.left),
                    "y": int(rect.top),
                    "w": int(rect.right - rect.left),
                    "h": int(rect.bottom - rect.top),
                },
                "is_foreground": int(hwnd) == int(fg_hwnd),
            })
        except Exception:
            # Any per-window failure is silently skipped so the whole
            # enumeration still returns a useful subset.
            pass
        return True

    user32.EnumWindows(WNDENUMPROC(_callback), 0)
    return results


def find_window(title_regex: str = "", handle: int = 0) -> int:
    """Resolve a window identifier to an HWND.

    Precedence:
        1. ``handle`` if > 0 and ``IsWindow`` confirms it is live.
        2. ``title_regex`` run via ``re.search(flags=IGNORECASE)`` against
           each enumerated window title. First match wins; enumeration
           order is OS-defined (typically z-order top-down).

    Returns 0 when nothing matches. Raises WindowCaptureError on invalid
    regex or bad handle (stale session).
    """
    _require_windows()

    import ctypes
    user32 = ctypes.windll.user32

    if handle:
        if not user32.IsWindow(int(handle)):
            raise WindowCaptureError(
                f"handle {handle!r} is not a live window (stale HWND?)")
        return int(handle)

    if not title_regex:
        return 0

    try:
        pattern = _re.compile(title_regex, _re.IGNORECASE)
    except _re.error as exc:
        raise WindowCaptureError(
            f"invalid title_regex {title_regex!r}: {exc}") from exc

    for w in list_windows_structured(
            include_hidden=False, include_minimized=True):
        if pattern.search(w["title"]):
            return int(w["handle"])
    return 0


def _capture_via_printwindow(hwnd: int) -> _Image.Image:
    """Ask the window to render its own content via PrintWindow.

    Uses the PW_RENDERFULLCONTENT flag (0x02) so that DWM-composed
    windows (most modern UWP / WPF surfaces) render cleanly. Raises
    WindowCaptureError if the window refuses; the caller handles
    fallback.
    """
    _require_windows()
    import ctypes
    import ctypes.wintypes
    import win32gui
    import win32ui
    import win32con  # noqa: F401  (kept for documentation; not strictly needed)

    user32 = ctypes.windll.user32
    if not user32.IsWindow(hwnd):
        raise WindowCaptureError(f"stale HWND {hwnd}")

    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    if width <= 0 or height <= 0:
        raise WindowCaptureError(
            f"window {hwnd} has non-positive bounds ({width}x{height})")

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    save_bitmap = win32ui.CreateBitmap()
    try:
        save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(save_bitmap)

        PW_RENDERFULLCONTENT = 0x00000002
        ret = ctypes.windll.user32.PrintWindow(
            hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)
        if not ret:
            raise WindowCaptureError(
                f"PrintWindow refused for HWND {hwnd} "
                "(likely a DirectX/OpenGL surface — region fallback will run)")

        bmpinfo = save_bitmap.GetInfo()
        bmpstr = save_bitmap.GetBitmapBits(True)
        pil = _Image.frombuffer(
            "RGB", (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
            bmpstr, "raw", "BGRX", 0, 1)
        return pil
    finally:
        try:
            win32gui.DeleteObject(save_bitmap.GetHandle())
        except Exception:
            pass
        try:
            save_dc.DeleteDC()
        except Exception:
            pass
        try:
            mfc_dc.DeleteDC()
        except Exception:
            pass
        try:
            win32gui.ReleaseDC(hwnd, hwnd_dc)
        except Exception:
            pass


def _capture_via_region(hwnd: int) -> _Image.Image:
    """Region-capture fallback via GetWindowRect + mss. Occluded pixels
    are NOT recovered — whatever is on top of the window leaks through.
    Used only when PrintWindow refuses or raises."""
    _require_windows()
    import ctypes
    import ctypes.wintypes

    from agent.screen_capture import ScreenCapture

    user32 = ctypes.windll.user32
    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    left = int(rect.left)
    top = int(rect.top)
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)

    # Clamp to screen bounds (maximized-window offsets can be slightly
    # negative on Windows 10/11; match the existing _screenshot_window
    # behaviour in mcp_server).
    if left < 0:
        width += left
        left = 0
    if top < 0:
        height += top
        top = 0
    if width <= 0 or height <= 0:
        raise WindowCaptureError(
            f"window {hwnd} clamped to non-positive region ({width}x{height})")

    cap = ScreenCapture(region=(left, top, width, height), resize_width=0)
    return cap.capture_as_pil()


def capture_window(
    title_regex: str = "",
    handle: int = 0,
    occluded: bool = True,
    resize_width: int = 0,
    save_path: str = "",
) -> _Image.Image:
    """Capture a single window and return its PIL image.

    The occluded=True default uses PrintWindow so that windows hidden
    behind others still yield their own content; pass occluded=False
    to use the region path unconditionally (faster for foreground
    windows).

    ``save_path`` is opt-in: the default empty string keeps the capture
    session-ephemeral. When set, the PNG is written after any resize but
    before privacy filtering so that the returned image and the saved
    file are the same bytes for caller verification.
    """
    _require_windows()

    hwnd = find_window(title_regex=title_regex, handle=handle)
    if not hwnd:
        raise WindowCaptureError(
            "no window matched "
            f"(title_regex={title_regex!r}, handle={handle!r})")

    if occluded:
        try:
            pil = _capture_via_printwindow(hwnd)
        except WindowCaptureError:
            # Fallback to region path so the caller still gets a frame.
            # Logging is handled by the calling MCP tool (keeps this
            # helper free of logging dependencies).
            pil = _capture_via_region(hwnd)
    else:
        pil = _capture_via_region(hwnd)

    if resize_width and pil.width and resize_width < pil.width:
        scale = resize_width / pil.width
        new_h = max(1, int(round(pil.height * scale)))
        pil = pil.resize((resize_width, new_h), _Image.Resampling.BICUBIC)

    if save_path:
        pil.save(save_path, format="PNG")

    return pil
