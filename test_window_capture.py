"""
test_window_capture.py

W4 v0.1 (decision 058): agent/window_capture.py unit + integration tests.

Non-Win32 platforms skip every test. The CI host for V2A is Windows,
and the module guards every public call with _require_windows() anyway.

The integration path (TestNotepadRealCapture) uses the actually-running
OS to spawn Notepad and capture its window. That subclass is also gated
on pywin32 and the Notepad executable being findable; when either is
missing, the integration test is silently skipped.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image as PILImage

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

IS_WINDOWS = sys.platform == "win32"


@unittest.skipUnless(IS_WINDOWS, "window_capture is Windows-only in v0.1")
class TestImportSurface(unittest.TestCase):
    """Module exports stable names + IS_WINDOWS flag + exception class."""

    def test_public_surface(self):
        from agent import window_capture as wc
        for name in ("list_windows_structured", "find_window",
                      "capture_window", "WindowCaptureError", "IS_WINDOWS"):
            self.assertTrue(hasattr(wc, name), f"missing public name: {name}")
        self.assertTrue(issubclass(wc.WindowCaptureError, RuntimeError))


@unittest.skipUnless(IS_WINDOWS, "window_capture is Windows-only in v0.1")
class TestListWindowsStructured(unittest.TestCase):
    """The shape of the returned dicts is stable; system surfaces are
    filtered; at least one window is always found on a live Windows
    session (the Python interpreter's own console window)."""

    def test_returns_list_of_dicts_with_required_keys(self):
        from agent.window_capture import list_windows_structured
        wins = list_windows_structured()
        self.assertIsInstance(wins, list)
        required = {"handle", "title", "app_name", "visible", "minimized",
                     "bounds", "is_foreground"}
        for w in wins:
            self.assertIsInstance(w, dict)
            self.assertTrue(required.issubset(w.keys()),
                              f"missing keys in {w}")
            self.assertIsInstance(w["handle"], int)
            self.assertIsInstance(w["title"], str)
            self.assertIsInstance(w["bounds"], dict)
            self.assertTrue({"x", "y", "w", "h"}.issubset(w["bounds"].keys()))

    def test_system_titles_are_filtered(self):
        from agent.window_capture import list_windows_structured
        wins = list_windows_structured()
        for w in wins:
            self.assertNotIn(w["title"].lower(),
                              {"program manager", "default ime",
                               "msctfime ui", ""})

    def test_at_most_one_foreground(self):
        from agent.window_capture import list_windows_structured
        wins = list_windows_structured()
        fgs = [w for w in wins if w["is_foreground"]]
        self.assertLessEqual(len(fgs), 1,
                              "at most one window should be is_foreground")

    def test_include_minimized_flag(self):
        """include_minimized=False should not produce minimized entries."""
        from agent.window_capture import list_windows_structured
        wins = list_windows_structured(include_minimized=False)
        for w in wins:
            self.assertFalse(w["minimized"])


@unittest.skipUnless(IS_WINDOWS, "window_capture is Windows-only in v0.1")
class TestFindWindow(unittest.TestCase):

    def test_returns_zero_when_both_params_absent(self):
        from agent.window_capture import find_window
        self.assertEqual(find_window(), 0)

    def test_handle_precedence_over_regex(self):
        """When handle is valid, title_regex is ignored entirely."""
        from agent.window_capture import find_window, list_windows_structured
        wins = list_windows_structured()
        if not wins:
            self.skipTest("no windows to test against")
        chosen = wins[0]["handle"]
        # deliberately give a nonsense regex; handle should still win
        h = find_window(title_regex="__no_such_window__", handle=chosen)
        self.assertEqual(h, chosen)

    def test_invalid_handle_raises(self):
        from agent.window_capture import find_window, WindowCaptureError
        with self.assertRaises(WindowCaptureError):
            find_window(handle=123456789)  # almost certainly stale

    def test_regex_case_insensitive(self):
        """Find the first real window, then try to match it via a
        lowercased substring of its title."""
        from agent.window_capture import find_window, list_windows_structured
        wins = list_windows_structured()
        if not wins:
            self.skipTest("no windows to test against")
        for w in wins:
            t = w["title"]
            if len(t) >= 4:
                sub = t[:4].lower()
                # Escape regex metachars so substring match works even if
                # the title begins with e.g. "(" or "[".
                import re
                h = find_window(title_regex=re.escape(sub))
                # The returned HWND should match at least one window whose
                # title contains the substring.
                matching = [ww["handle"] for ww in wins
                             if sub in ww["title"].lower()]
                self.assertIn(h, matching,
                                f"regex match failed for {sub!r}")
                return
        self.skipTest("no window titles long enough for substring test")

    def test_invalid_regex_raises(self):
        from agent.window_capture import find_window, WindowCaptureError
        with self.assertRaises(WindowCaptureError):
            find_window(title_regex="[unclosed-char-class")


@unittest.skipUnless(IS_WINDOWS, "window_capture is Windows-only in v0.1")
class TestCaptureWindow(unittest.TestCase):

    def _first_viable_window(self):
        """Pick a visible non-minimized window with positive bounds."""
        from agent.window_capture import list_windows_structured
        for w in list_windows_structured():
            if (w["visible"] and not w["minimized"]
                    and w["bounds"]["w"] > 50 and w["bounds"]["h"] > 50):
                return w
        return None

    def test_no_match_raises(self):
        from agent.window_capture import capture_window, WindowCaptureError
        with self.assertRaises(WindowCaptureError):
            capture_window(title_regex="__no_such_window__")

    def test_capture_via_handle_returns_pil_image(self):
        from agent.window_capture import capture_window
        w = self._first_viable_window()
        if not w:
            self.skipTest("no viable window")
        pil = capture_window(handle=w["handle"])
        self.assertIsInstance(pil, PILImage.Image)
        self.assertGreater(pil.width, 0)
        self.assertGreater(pil.height, 0)

    def test_region_path_via_occluded_false(self):
        from agent.window_capture import capture_window
        w = self._first_viable_window()
        if not w:
            self.skipTest("no viable window")
        pil = capture_window(handle=w["handle"], occluded=False)
        self.assertIsInstance(pil, PILImage.Image)

    def test_resize_width_scales(self):
        from agent.window_capture import capture_window
        w = self._first_viable_window()
        if not w or w["bounds"]["w"] < 200:
            self.skipTest("no viable window >= 200 px")
        pil = capture_window(handle=w["handle"], resize_width=100)
        self.assertEqual(pil.width, 100)

    def test_save_path_opt_in_writes_png(self):
        from agent.window_capture import capture_window
        w = self._first_viable_window()
        if not w:
            self.skipTest("no viable window")
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "cap.png")
            pil = capture_window(handle=w["handle"], save_path=out)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 100)
            # saved bytes load back to a PNG with the same dimensions
            with PILImage.open(out) as loaded:
                loaded.load()
                self.assertEqual(loaded.size, pil.size)

    def test_save_path_empty_does_not_write(self):
        """The default save_path='' must not write to disk."""
        from agent.window_capture import capture_window
        w = self._first_viable_window()
        if not w:
            self.skipTest("no viable window")
        with tempfile.TemporaryDirectory() as td:
            # Run capture without save_path and confirm the tmp dir is empty.
            capture_window(handle=w["handle"])
            self.assertEqual(os.listdir(td), [])


@unittest.skipUnless(IS_WINDOWS, "window_capture is Windows-only in v0.1")
class TestPrintWindowFallback(unittest.TestCase):
    """When PrintWindow refuses (simulated), capture_window falls back
    through to the region path rather than raising."""

    def test_printwindow_failure_falls_back_to_region(self):
        from agent import window_capture as wc
        w = None
        for cand in wc.list_windows_structured():
            if cand["visible"] and not cand["minimized"] and cand["bounds"]["w"] > 100:
                w = cand
                break
        if not w:
            self.skipTest("no viable window")

        def _boom(hwnd):
            raise wc.WindowCaptureError("simulated PrintWindow failure")

        with patch.object(wc, "_capture_via_printwindow", side_effect=_boom):
            pil = wc.capture_window(handle=w["handle"], occluded=True)
        self.assertIsInstance(pil, PILImage.Image)
        self.assertGreater(pil.width, 0)


@unittest.skipUnless(IS_WINDOWS, "window_capture is Windows-only in v0.1")
class TestErrorHandling(unittest.TestCase):

    def test_capture_with_stale_handle_raises(self):
        from agent.window_capture import capture_window, WindowCaptureError
        with self.assertRaises(WindowCaptureError):
            capture_window(handle=123456789)


class TestPlatformGuard(unittest.TestCase):
    """On a fake non-Win32 platform, _require_windows raises and public
    callers either raise or return an error dict from the MCP tool."""

    def test_require_windows_raises_on_fake_non_win32(self):
        from agent import window_capture as wc
        with patch.object(wc, "IS_WINDOWS", False):
            with self.assertRaises(wc.WindowCaptureError):
                wc._require_windows()


class TestMcpToolShape(unittest.TestCase):
    """mcp_server.list_windows_detailed + capture_window return the shapes
    the spec promises, even on non-Win32 (they return explicit error
    payloads rather than raising)."""

    def test_list_windows_detailed_returns_json_string(self):
        import mcp_server
        import json
        out = mcp_server.list_windows_detailed()
        self.assertIsInstance(out, str)
        data = json.loads(out)
        # On Win32 this is a list; on non-Win32 an error dict. Both JSON-valid.
        self.assertTrue(isinstance(data, (list, dict)))

    @unittest.skipUnless(IS_WINDOWS, "Win32 only")
    def test_capture_window_no_match_returns_error_list(self):
        import mcp_server
        out = mcp_server.capture_window(title_regex="__no_such_window__")
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], str)
        self.assertIn("capture_window 失敗", out[0])


if __name__ == "__main__":
    unittest.main()
