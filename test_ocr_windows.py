"""
test_ocr_windows.py
agent.ocr_windows の後方互換 + 新規 bbox API の単体テスト。
"""
import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

from PIL import Image

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent import ocr_windows


class TestBackwardCompat(unittest.TestCase):
    """既存 API シグネチャが変わっていないことを保証する。"""

    def test_ocr_image_signature_unchanged(self):
        import inspect
        sig = inspect.signature(ocr_windows.ocr_image)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["image", "timeout", "preprocess"])

    def test_ocr_region_signature_unchanged(self):
        import inspect
        sig = inspect.signature(ocr_windows.ocr_region)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["monitor", "left", "top", "width", "height"])

    def test_ocr_image_returns_str(self):
        img = Image.new("RGB", (200, 100), color="white")
        if sys.platform != "win32":
            self.assertEqual(ocr_windows.ocr_image(img), "")
        else:
            result = ocr_windows.ocr_image(img)
            self.assertIsInstance(result, str)


class TestBboxAPI(unittest.TestCase):
    """新規 ocr_image_with_bboxes の契約テスト。"""

    def test_non_windows_returns_empty(self):
        img = Image.new("RGB", (200, 100), color="white")
        if sys.platform != "win32":
            self.assertEqual(ocr_windows.ocr_image_with_bboxes(img), [])

    @unittest.skipUnless(sys.platform == "win32", "Windows only")
    def test_returns_list_of_dicts(self):
        img = Image.new("RGB", (200, 100), color="white")
        result = ocr_windows.ocr_image_with_bboxes(img)
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIn("text", item)
            self.assertIn("bbox", item)
            self.assertIn("x", item["bbox"])
            self.assertIn("y", item["bbox"])
            self.assertIn("w", item["bbox"])
            self.assertIn("h", item["bbox"])

    @unittest.skipUnless(sys.platform == "win32", "Windows only")
    def test_scale_back_from_preprocess_upscale(self):
        """
        小画像 (<1000px wide) は _preprocess で 2x される。
        返却される bbox 座標は入力画像の座標系（÷2 で元に戻す）でなければならない。
        """
        img = Image.new("RGB", (500, 200), color="white")  # < 1000 → 2x upscale
        fake_ps_output = json.dumps([
            {"text": "A", "x": 100.0, "y": 50.0, "w": 40.0, "h": 30.0}
        ])

        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = fake_ps_output.encode("utf-8")
        with patch.object(ocr_windows.subprocess, "run", return_value=fake_result):
            words = ocr_windows.ocr_image_with_bboxes(img, preprocess=True)
        self.assertEqual(len(words), 1)
        b = words[0]["bbox"]
        self.assertEqual(b["x"], 50.0)
        self.assertEqual(b["y"], 25.0)
        self.assertEqual(b["w"], 20.0)
        self.assertEqual(b["h"], 15.0)

    @unittest.skipUnless(sys.platform == "win32", "Windows only")
    def test_no_scale_for_large_image(self):
        """1000px 以上の画像は upscale されない → 座標はそのまま返る。"""
        img = Image.new("RGB", (1200, 400), color="white")
        fake_ps_output = json.dumps([
            {"text": "B", "x": 300.0, "y": 100.0, "w": 80.0, "h": 40.0}
        ])
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = fake_ps_output.encode("utf-8")
        with patch.object(ocr_windows.subprocess, "run", return_value=fake_result):
            words = ocr_windows.ocr_image_with_bboxes(img, preprocess=True)
        self.assertEqual(len(words), 1)
        b = words[0]["bbox"]
        self.assertEqual(b["x"], 300.0)
        self.assertEqual(b["y"], 100.0)

    @unittest.skipUnless(sys.platform == "win32", "Windows only")
    def test_single_word_json_dict_coerced_to_list(self):
        """PowerShell ConvertTo-Json は 1 要素だと object 単体を返すため list 化されること。"""
        img = Image.new("RGB", (1200, 400), color="white")
        fake_ps_output = json.dumps(
            {"text": "Solo", "x": 10.0, "y": 10.0, "w": 20.0, "h": 20.0}
        )
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = fake_ps_output.encode("utf-8")
        with patch.object(ocr_windows.subprocess, "run", return_value=fake_result):
            words = ocr_windows.ocr_image_with_bboxes(img, preprocess=True)
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0]["text"], "Solo")

    @unittest.skipUnless(sys.platform == "win32", "Windows only")
    def test_invalid_json_returns_empty(self):
        img = Image.new("RGB", (1200, 400), color="white")
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = b"not valid json"
        with patch.object(ocr_windows.subprocess, "run", return_value=fake_result):
            self.assertEqual(
                ocr_windows.ocr_image_with_bboxes(img, preprocess=True), []
            )

    @unittest.skipUnless(sys.platform == "win32", "Windows only")
    def test_subprocess_failure_returns_empty(self):
        img = Image.new("RGB", (1200, 400), color="white")
        fake_result = MagicMock()
        fake_result.returncode = 1
        fake_result.stdout = b""
        with patch.object(ocr_windows.subprocess, "run", return_value=fake_result):
            self.assertEqual(
                ocr_windows.ocr_image_with_bboxes(img, preprocess=True), []
            )


if __name__ == "__main__":
    unittest.main()
