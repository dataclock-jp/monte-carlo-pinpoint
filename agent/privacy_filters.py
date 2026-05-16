"""Privacy filters for Video2AI — capabilities, not policy.

Video2AI provides filter *capabilities* (tools).
Neo decides filter *policy* (when to apply).

Filters:
  - blur_faces: Detect and Gaussian-blur human faces
  - redact_text: Detect and mask PII text (credit cards, phone numbers, emails)

All filters operate on PIL Images and return a filtered copy.
The original is never modified.
"""

import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage


# ============================================================================
# Face Detection + Blur
# ============================================================================

# OpenCV DNN face detector (bundled with opencv-python)
_face_net = None
_face_net_loaded = False

# Haar cascade as fallback
_face_cascade = None
_face_cascade_loaded = False


def _load_face_detector_dnn():
    """Load OpenCV DNN face detector (Caffe model)."""
    global _face_net, _face_net_loaded
    if _face_net_loaded:
        return _face_net

    _face_net_loaded = True
    # Try to find the bundled model
    data_dir = Path(cv2.data.haarcascades).parent
    prototxt = data_dir / "deploy.prototxt"
    caffemodel = data_dir / "res10_300x300_ssd_iter_140000_fp16.caffemodel"

    if prototxt.exists() and caffemodel.exists():
        _face_net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
        return _face_net
    return None


def _load_face_detector_haar():
    """Load Haar cascade face detector (always available with OpenCV)."""
    global _face_cascade, _face_cascade_loaded
    if _face_cascade_loaded:
        return _face_cascade

    _face_cascade_loaded = True
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if Path(cascade_path).exists():
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        if _face_cascade.empty():
            _face_cascade = None
    return _face_cascade


def _detect_faces_dnn(img_bgr: np.ndarray,
                      confidence_threshold: float = 0.5) -> list[tuple]:
    """Detect faces using DNN. Returns list of (x, y, w, h)."""
    net = _load_face_detector_dnn()
    if net is None:
        return []

    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(img_bgr, 1.0, (300, 300),
                                  (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > confidence_threshold:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)
            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces


def _detect_faces_haar(img_bgr: np.ndarray) -> list[tuple]:
    """Detect faces using Haar cascade. Returns list of (x, y, w, h)."""
    cascade = _load_face_detector_haar()
    if cascade is None:
        return []

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                     minNeighbors=5, minSize=(30, 30))
    return [(x, y, w, h) for x, y, w, h in faces]


def detect_faces(img_bgr: np.ndarray,
                 confidence: float = 0.5) -> list[tuple]:
    """Detect faces using best available method.

    Returns list of (x, y, w, h) bounding boxes.
    Tries DNN first, falls back to Haar cascade.
    """
    faces = _detect_faces_dnn(img_bgr, confidence)
    if not faces:
        faces = _detect_faces_haar(img_bgr)
    return faces


def blur_faces(pil_img: PILImage.Image,
               blur_strength: int = 51,
               confidence: float = 0.5) -> PILImage.Image:
    """Detect and Gaussian-blur all faces in the image.

    Args:
        pil_img: Input PIL image (not modified)
        blur_strength: Gaussian kernel size (must be odd, higher = more blur)
        confidence: DNN detection confidence threshold

    Returns:
        New PIL image with faces blurred. Returns copy even if no faces found.
    """
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    faces = detect_faces(img_bgr, confidence)

    if not faces:
        return pil_img.copy()

    result = img_bgr.copy()
    # Ensure odd kernel size
    k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1

    for (x, y, w, h) in faces:
        # Expand region slightly for better coverage
        pad = int(max(w, h) * 0.2)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(result.shape[1], x + w + pad)
        y2 = min(result.shape[0], y + h + pad)

        roi = result[y1:y2, x1:x2]
        result[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 30)

    return PILImage.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


# ============================================================================
# PII Text Redaction
# ============================================================================

# Common PII patterns
_PII_PATTERNS = [
    # Credit card numbers (13-19 digits, optionally separated by spaces/dashes)
    (r'\b(?:\d[ -]*?){13,19}\b', '[CARD]'),
    # Japanese phone numbers
    (r'\b0\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{3,4}\b', '[PHONE]'),
    # International phone numbers
    (r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b', '[PHONE]'),
    # Email addresses
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL]'),
    # Japanese My Number (12 digits)
    (r'\b\d{4}\s?\d{4}\s?\d{4}\b', '[MYNUMBER]'),
    # IP addresses
    (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]'),
]

_compiled_pii = [(re.compile(p), r) for p, r in _PII_PATTERNS]


def redact_text(text: str) -> str:
    """Redact PII patterns from text.

    Replaces credit card numbers, phone numbers, email addresses,
    Japanese My Number, and IP addresses with placeholder tokens.

    Args:
        text: Input text

    Returns:
        Text with PII replaced by tokens like [CARD], [PHONE], [EMAIL]
    """
    result = text
    for pattern, replacement in _compiled_pii:
        result = pattern.sub(replacement, result)
    return result


def redact_text_regions(pil_img: PILImage.Image,
                        ocr_func=None) -> PILImage.Image:
    """Detect and mask PII text regions in an image.

    Uses OCR to find text, checks for PII patterns, and blacks out
    the regions containing sensitive text.

    Args:
        pil_img: Input PIL image
        ocr_func: Optional OCR function(pil_img) -> list of
                  {"text": str, "bounds": (x, y, w, h)} dicts.
                  If None, returns the image unchanged.

    Returns:
        New PIL image with PII regions blacked out.
    """
    if ocr_func is None:
        return pil_img.copy()

    try:
        regions = ocr_func(pil_img)
    except Exception:
        return pil_img.copy()

    if not regions:
        return pil_img.copy()

    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    result = img_bgr.copy()

    for region in regions:
        text = region.get("text", "")
        bounds = region.get("bounds")
        if not bounds:
            continue

        # Check if text contains PII
        redacted = redact_text(text)
        if redacted != text:
            x, y, w, h = bounds
            # Black out the region
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 0), -1)

    return PILImage.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


# ============================================================================
# Combined Pipeline
# ============================================================================

def apply_privacy_filters(
    pil_img: PILImage.Image,
    blur_faces_enabled: bool = False,
    redact_text_enabled: bool = False,
    ocr_func=None,
) -> PILImage.Image:
    """Apply all enabled privacy filters to an image.

    This is the main entry point for privacy filtering.
    Filters are applied in order: face blur first, then text redaction.

    Args:
        pil_img: Input PIL image (not modified)
        blur_faces_enabled: Enable face detection + blur
        redact_text_enabled: Enable PII text detection + masking
        ocr_func: OCR function for text redaction (optional)

    Returns:
        New PIL image with all enabled filters applied.
        Returns a copy even if no filters are enabled.
    """
    result = pil_img

    if blur_faces_enabled:
        result = blur_faces(result)

    if redact_text_enabled:
        result = redact_text_regions(result, ocr_func)

    if result is pil_img:
        result = pil_img.copy()

    return result
