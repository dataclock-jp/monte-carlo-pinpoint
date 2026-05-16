"""grid_find / pinpoint のコアロジック（MCP 非依存）

mcp_server.py から画面キャプチャ非依存の関数を抽出。
ベンチマークで画像ファイルに対して実行するために使用。
"""

import base64
import io
import logging
import math
import os
import random
import re
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image as PILImage, ImageDraw, ImageFont


# ============================================================================
# API key / VLM call
# ============================================================================

_USE_CLI = False   # Set True to use claude CLI (subscription) instead of API

def get_api_key() -> str:
    import json
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        p = Path.home() / ".claude.json"
        if p.exists():
            cfg = json.loads(p.read_text(encoding="utf-8"))
            key = (cfg.get("mcpServers", {})
                   .get("video2ai-desktop", {})
                   .get("env", {})
                   .get("ANTHROPIC_API_KEY", ""))
    return key


def _vlm_call_cli(b64_image: str, system: str, user: str,
                  model: str = "sonnet") -> str | None:
    """VLM call via claude CLI subprocess.

    Forces Claude Code to fall back to OAuth subscription (Max/Pro) by
    stripping ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN from the child env.
    Without this, claude CLI auth precedence picks API key over OAuth and
    every benchmark call hits Anthropic API billing instead of subscription.
    """
    img_data = base64.b64decode(b64_image)
    fd, temp_path = tempfile.mkstemp(suffix='.jpg',
                                     dir=str(Path(__file__).parent))
    try:
        os.write(fd, img_data)
        os.close(fd)

        img_path = temp_path.replace("\\", "/")

        # User prompt: read image + question
        prompt = (
            f"Read the image at {img_path} using the Read tool, "
            f"then answer.\n\n{user}"
        )

        # System context via --append-system-prompt (preserves default prompt)
        sys_prompt = (
            f"{system}\n\n"
            "CRITICAL: Reply with ONLY the short answer code/value. "
            "No explanation, no commentary, no description of what you see."
        )

        # Map full model names to CLI aliases
        cli_model = model
        if "sonnet" in model:
            cli_model = "sonnet"
        elif "opus" in model:
            cli_model = "opus"
        elif "haiku" in model:
            cli_model = "haiku"

        # Subscription-only enforcement: strip API-key auth vars from child
        # env so claude CLI falls back to OAuth (see docstring above).
        subprocess_env = os.environ.copy()
        subprocess_env.pop("ANTHROPIC_API_KEY", None)
        subprocess_env.pop("ANTHROPIC_AUTH_TOKEN", None)

        result = subprocess.run(
            ["claude", "-p", prompt,
             "--append-system-prompt", sys_prompt,
             "--output-format", "text",
             "--tools", "Read",
             "--allowedTools", "Read",
             "--model", cli_model,
             "--no-session-persistence",
             "--setting-sources", "user"],
            capture_output=True, text=True, timeout=180,
            encoding="utf-8",
            env=subprocess_env,
        )

        if result.returncode != 0:
            logging.warning("CLI error (rc=%d): %s",
                            result.returncode, result.stderr[:300])
            return None

        output = result.stdout.strip()
        if not output:
            return None

        # Take last non-empty line (skip any preamble from tool use)
        lines = [l.strip() for l in output.splitlines() if l.strip()]
        answer = lines[-1] if lines else None
        logging.info("CLI raw output: %r -> answer: %r", output[:300], answer)
        return answer

    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def vlm_call(b64_image: str, system: str, user: str, api_key: str,
             model: str, max_tokens: int = 10,
             _max_retries: int = 8) -> str | None:
    """Anthropic Messages API call with rate-limit retry.
    When _USE_CLI is True, delegates to claude CLI (subscription)."""
    if _USE_CLI:
        return _vlm_call_cli(b64_image, system, user, model)

    import requests as _requests

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": system,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": b64_image}},
                {"type": "text", "text": user},
            ],
        }],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    for attempt in range(_max_retries):
        try:
            resp = _requests.post(
                "https://api.anthropic.com/v1/messages",
                json=payload, headers=headers, timeout=60,
            )
            if resp.status_code == 200:
                return resp.json()["content"][0]["text"].strip()

            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    wait = float(retry_after)
                else:
                    wait = min(2 ** attempt * 5, 120)
                logging.warning("Rate limit (429), waiting %.0fs (attempt %d/%d)",
                                wait, attempt + 1, _max_retries)
                time.sleep(wait)
                continue

            if resp.status_code == 529:
                wait = min(2 ** attempt * 10, 120)
                logging.warning("Overloaded (529), waiting %.0fs", wait)
                time.sleep(wait)
                continue

            logging.warning("VLM API %s: %s", resp.status_code, resp.text[:200])
            return None

        except _requests.exceptions.Timeout:
            logging.warning("VLM API timeout (attempt %d/%d)", attempt + 1, _max_retries)
            time.sleep(5)
            continue
        except Exception as e:
            logging.warning("VLM API error: %s", e)
            return None

    logging.warning("VLM API: max retries (%d) exhausted", _max_retries)
    return None


def to_base64(pil_img: PILImage.Image, quality: int = 80) -> str:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ============================================================================
# Grid Find (from mcp_server.py)
# ============================================================================

def grid_find_parse_response(text: str, num_lines: int) -> dict:
    text = text.strip().upper()
    m = re.search(r"\bL\s*(\d+)\b", text)
    if m:
        n = int(m.group(1))
        if 1 <= n <= num_lines:
            return {"type": "exact", "line": n}
    m = re.search(r"\bB\s*(\d+)\s*[-,]\s*(\d+)\b", text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if 0 <= a and b <= num_lines + 1 and a < b:
            return {"type": "between", "line_a": a, "line_b": b}
    if "NONE" in text:
        return {"type": "not_found"}
    return {"type": "not_found"}


def grid_find_line_pos(line_num: int, lo: int, hi: int, num_lines: int) -> int:
    if line_num <= 0:
        return lo
    if line_num > num_lines:
        return hi
    margin = int((hi - lo) * 0.02)
    inner_lo = lo + margin
    inner_hi = hi - margin
    if num_lines == 1:
        return (inner_lo + inner_hi) // 2
    return inner_lo + int((line_num - 1) * (inner_hi - inner_lo) / (num_lines - 1))


def grid_find_draw_lines(pil_img: PILImage.Image, axis: str,
                         lo: int, hi: int, num_lines: int,
                         crosshair_pos: int = -1) -> PILImage.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    line_color = (0, 220, 220) if axis == "x" else (220, 0, 220)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    # Overlay to dim outside search area
    overlay = PILImage.new("RGBA", img.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    if axis == "x":
        if lo > 0:
            ov_draw.rectangle([0, 0, lo, h], fill=(0, 0, 0, 60))
        if hi < w:
            ov_draw.rectangle([hi, 0, w, h], fill=(0, 0, 0, 60))
    else:
        if lo > 0:
            ov_draw.rectangle([0, 0, w, lo], fill=(0, 0, 0, 60))
        if hi < h:
            ov_draw.rectangle([0, hi, w, h], fill=(0, 0, 0, 60))
    img = PILImage.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    for i in range(1, num_lines + 1):
        pos = grid_find_line_pos(i, lo, hi, num_lines)
        label = str(i)
        if axis == "x":
            draw.line([(pos, 0), (pos, h)], fill=line_color, width=2)
            bbox = font.getbbox(label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            lx, ly = pos - tw // 2, 4
            draw.rectangle([lx - 2, ly - 1, lx + tw + 2, ly + th + 1], fill=(0, 0, 0))
            draw.text((lx, ly), label, fill=(255, 255, 255), font=font)
        else:
            draw.line([(0, pos), (w, pos)], fill=line_color, width=2)
            bbox = font.getbbox(label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            lx, ly = 4, pos - th // 2
            draw.rectangle([lx - 2, ly - 1, lx + tw + 2, ly + th + 1], fill=(0, 0, 0))
            draw.text((lx, ly), label, fill=(255, 255, 255), font=font)

    if crosshair_pos >= 0:
        ch_color = (255, 200, 0)
        if axis == "x":
            draw.line([(0, crosshair_pos), (w, crosshair_pos)], fill=ch_color, width=1)
        else:
            draw.line([(crosshair_pos, 0), (crosshair_pos, h)], fill=ch_color, width=1)
    return img


def grid_find_ask_llm(b64_image: str, target: str, axis: str,
                      num_lines: int, api_key: str, model: str) -> dict:
    direction = "vertical" if axis == "x" else "horizontal"
    axis_label = "X (horizontal position)" if axis == "x" else "Y (vertical position)"
    edge_lo = "left" if axis == "x" else "above"
    edge_hi = "right" if axis == "x" else "below"

    system = (
        "You locate UI elements on screen. "
        f"The image has numbered {direction} lines (1-{num_lines}). "
        f"Find which line is closest to the target's {axis_label}. "
        "Reply with ONLY one of these codes, nothing else:\n"
        "L5 = line 5 crosses target\n"
        "B3-4 = target is between lines 3 and 4\n"
        f"B0-1 = target is {edge_lo} of line 1\n"
        f"B{num_lines}-{num_lines + 1} = target is {edge_hi} of line {num_lines}\n"
        "NONE = target not visible\n"
        "Output ONLY the code (e.g. L5 or B3-4). No words, no explanation."
    )
    user_msg = f"Target: {target}\nAnswer with code only:"
    text = vlm_call(b64_image, system, user_msg, api_key, model, max_tokens=10)
    if not text:
        return {"type": "not_found", "_raw": ""}
    parsed = grid_find_parse_response(text, num_lines)
    parsed["_raw"] = text
    return parsed


def grid_search_axis(pil_img: PILImage.Image, target: str, axis: str,
                     num_lines: int, precision_px: int,
                     api_key: str, model: str,
                     crosshair_pos: int = -1,
                     query_counter: list | None = None) -> tuple[int, list[str]]:
    """1 axis binary search with recursive zoom. Returns (coord, log).
    query_counter: if provided, [count] is incremented per VLM call."""
    shot_size = pil_img.width if axis == "x" else pil_img.height
    lo, hi = 0, shot_size
    steps = []
    # Claude Sonnet silently downsamples 1:1 images above ~1092x1092
    # (per the "1568px rule"). 1024 is the no-loss safe cap.
    ZOOM_UPSCALE = 1024
    ZOOM_THRESHOLD = shot_size // 3

    for iteration in range(8):
        interval = hi - lo
        if interval <= precision_px:
            break

        # Zoom mode
        if interval < ZOOM_THRESHOLD and interval > 0:
            ctx = max(interval * 3, 100)
            crop_lo = max(0, lo - ctx)
            crop_hi = min(shot_size, hi + ctx)
            if axis == "x":
                cropped = pil_img.crop((crop_lo, 0, crop_hi, pil_img.height))
            else:
                cropped = pil_img.crop((0, crop_lo, pil_img.width, crop_hi))
            cw, ch = cropped.size
            if axis == "x":
                sf = ZOOM_UPSCALE / cw
                new_size = (ZOOM_UPSCALE, max(1, int(ch * sf)))
            else:
                sf = ZOOM_UPSCALE / ch
                new_size = (max(1, int(cw * sf)), ZOOM_UPSCALE)
            # Clamp to API max image dimension (8000px)
            max_dim = max(new_size)
            if max_dim > 7900:
                clamp_sf = 7900 / max_dim
                new_size = (max(1, int(new_size[0] * clamp_sf)),
                            max(1, int(new_size[1] * clamp_sf)))
                sf = sf * clamp_sf
            work_img = cropped.resize(new_size, PILImage.Resampling.BICUBIC)
            zoom_lo = int((lo - crop_lo) * sf)
            zoom_hi = int((hi - crop_lo) * sf)
            zoom_size = new_size[0] if axis == "x" else new_size[1]
            zoom_crosshair = -1
            if crosshair_pos >= 0:
                if axis == "x":
                    zoom_crosshair = int(crosshair_pos * sf)
                else:
                    zoom_crosshair = int((crosshair_pos - crop_lo) * sf) if axis == "y" else -1

            annotated = grid_find_draw_lines(
                work_img, axis, zoom_lo, zoom_hi, num_lines, zoom_crosshair)
            b64 = to_base64(annotated)
            if query_counter is not None:
                query_counter[0] += 1
            result = grid_find_ask_llm(b64, target, axis, num_lines, api_key, model)
            raw = result.get("_raw", "?")

            if result["type"] == "exact":
                zoom_pos = grid_find_line_pos(result["line"], zoom_lo, zoom_hi, num_lines)
                pos = crop_lo + int(zoom_pos / sf)
                steps.append(f"  {axis} step{iteration+1} [zoom]: L{result['line']} -> {pos} (raw={raw!r})")
                return pos, steps
            elif result["type"] == "between":
                z_lo = grid_find_line_pos(result["line_a"], zoom_lo, zoom_hi, num_lines)
                z_hi = grid_find_line_pos(result["line_b"], zoom_lo, zoom_hi, num_lines)
                lo = crop_lo + int(z_lo / sf)
                hi = crop_lo + int(z_hi / sf)
                steps.append(f"  {axis} step{iteration+1} [zoom]: B{result['line_a']}-{result['line_b']} -> [{lo}..{hi}] (raw={raw!r})")
            else:
                steps.append(f"  {axis} step{iteration+1} [zoom]: NOT FOUND (raw={raw!r})")
                return -1, steps
        else:
            # Normal mode
            annotated = grid_find_draw_lines(
                pil_img, axis, lo, hi, num_lines, crosshair_pos)
            b64 = to_base64(annotated)
            if query_counter is not None:
                query_counter[0] += 1
            result = grid_find_ask_llm(b64, target, axis, num_lines, api_key, model)
            raw = result.get("_raw", "?")

            if result["type"] == "exact":
                ln = result["line"]
                if iteration == 0:
                    # First iteration: don't trust exact — expand to adjacent lines
                    # and force a zoom refinement pass
                    adj_lo = max(1, ln - 1)
                    adj_hi = min(num_lines, ln + 1) + 1  # +1 for B notation
                    new_lo = grid_find_line_pos(adj_lo, lo, hi, num_lines)
                    new_hi = grid_find_line_pos(adj_hi - 1, lo, hi, num_lines)
                    steps.append(f"  {axis} step{iteration+1}: L{ln}->B{adj_lo}-{adj_hi-1} (force zoom, raw={raw!r})")
                    lo, hi = new_lo, new_hi
                else:
                    pos = grid_find_line_pos(ln, lo, hi, num_lines)
                    steps.append(f"  {axis} step{iteration+1}: L{ln} -> {pos} (raw={raw!r})")
                    return pos, steps
            elif result["type"] == "between":
                new_lo = grid_find_line_pos(result["line_a"], lo, hi, num_lines)
                new_hi = grid_find_line_pos(result["line_b"], lo, hi, num_lines)
                steps.append(f"  {axis} step{iteration+1}: B{result['line_a']}-{result['line_b']} -> [{new_lo}..{new_hi}] (raw={raw!r})")
                lo, hi = new_lo, new_hi
            else:
                steps.append(f"  {axis} step{iteration+1}: NOT FOUND (raw={raw!r})")
                return -1, steps

    pos = (lo + hi) // 2
    steps.append(f"  {axis} converged: {pos} (interval={hi-lo}px)")
    return pos, steps


# ============================================================================
# Pinpoint (Phase 1 + 1.5 + 2)
# ============================================================================

PINPOINT_COLORS_DEFAULT = [
    (255, 0, 0), (0, 200, 0), (0, 100, 255), (255, 200, 0),
    (255, 0, 200), (0, 220, 220), (255, 128, 0),
]

# Dark-background-friendly colors (high brightness)
PINPOINT_COLORS_BRIGHT = [
    (255, 100, 100), (100, 255, 100), (100, 200, 255), (255, 255, 100),
    (255, 150, 255), (100, 255, 255), (255, 200, 100),
]

# Light-background-friendly colors (high saturation, medium brightness)
PINPOINT_COLORS_DARK = [
    (200, 0, 0), (0, 150, 0), (0, 0, 200), (180, 140, 0),
    (180, 0, 150), (0, 150, 150), (200, 80, 0),
]

# Legacy alias
PINPOINT_COLORS = PINPOINT_COLORS_DEFAULT


def _adaptive_dot_colors(pil_region: "PILImage.Image", num_colors: int = 7) -> list[tuple]:
    """Pick dot colors that contrast with the background.

    Analyzes the average brightness and dominant hues of the crop region,
    then selects a color palette that maximizes visibility.
    """
    import numpy as np
    arr = np.array(pil_region)
    if arr.size == 0:
        return PINPOINT_COLORS_DEFAULT[:num_colors]

    avg_brightness = arr.mean()

    if avg_brightness < 80:
        # Dark background -> use bright colors
        return PINPOINT_COLORS_BRIGHT[:num_colors]
    elif avg_brightness > 200:
        # Light/white background -> use saturated dark colors
        return PINPOINT_COLORS_DARK[:num_colors]
    else:
        # Medium -> use default high-saturation colors
        return PINPOINT_COLORS_DEFAULT[:num_colors]


def pinpoint_declare_anchor(b64_image: str, target: str,
                             api_key: str, model: str) -> tuple[str, bool]:
    """Phase 1.5: Target visibility check + exact pixel declaration.

    GR-0 Anchor Validity Check: catches cascading hallucination at
    declaration time when Phase-1 lands in the wrong neighborhood.
    Asking "Is the target visible?" explicitly gives the VLM a structured
    escape hatch (ABSENT) before it commits to describing a non-existent
    target, breaking the faithfulness-bias chain at its source.

    Returns:
        (anchor_text, visible)
        - visible=True:  target is visible, anchor_text describes the pixel
        - visible=False: target is absent, anchor_text describes what is seen
    """
    system = (
        "You are a pixel-precision visual inspector. "
        "You will be asked whether a target is visible in a zoomed image. "
        "If it is, declare EXACTLY which pixel you consider the correct "
        "target point -- not 'the gear icon' but 'the exact center pixel "
        "of the gear icon's hub circle'. "
        "If the target is NOT clearly present in the image, you must "
        "explicitly report its absence instead of inventing a location. "
        "This declaration is the strict standard for all subsequent judgments."
    )
    user = (
        f"Target: {target}\n\n"
        "Is this target clearly visible in the image? "
        "Respond in this EXACT format:\n"
        "VISIBLE: <one sentence specifying the exact target pixel "
        "(center, edge, or feature point)>\n"
        "or\n"
        "ABSENT: <one sentence describing what you see instead>\n\n"
        "Start your response with either 'VISIBLE:' or 'ABSENT:' -- "
        "no other prefix. One sentence only."
    )
    text = vlm_call(b64_image, system, user, api_key, model, max_tokens=120)
    if not text:
        return (target, True)  # API failure -> fall back to original target
    text = text.strip()
    upper = text.upper()
    if upper.startswith("ABSENT"):
        desc = text.split(":", 1)[1].strip() if ":" in text else text
        return (desc, False)
    if upper.startswith("VISIBLE"):
        desc = text.split(":", 1)[1].strip() if ":" in text else text
        return (desc, True)
    # Refusal patterns (legacy path): treat as ABSENT
    if (upper.startswith("I CANNOT") or upper.startswith("I DO NOT")
            or upper.startswith("I CAN'T") or upper.startswith("I'M UNABLE")):
        return (text, False)
    # Unstructured response: treat as visible (backwards compatible)
    return (text, True)


def pinpoint_ask_closest(b64_image: str, anchor: str, num_dots: int,
                         api_key: str, model: str) -> int | None:
    """Ask VLM which dot is CLOSEST to the target (always returns a dot)."""
    system = (
        "You are a pixel-precision visual inspector. "
        f"The image shows a highly zoomed-in area with {num_dots} small "
        "numbered colored dots (tiny circles with number labels). "
        "You previously declared the exact target point as:\n"
        f'  "{anchor}"\n\n'
        "TASK: Identify which numbered dot is CLOSEST to the declared "
        "target point. There is always a closest dot — you must pick one.\n\n"
        "Reply with ONLY: D<number> (e.g. D3)\n"
        "Output ONLY the code. No explanation."
    )
    text = vlm_call(b64_image, system,
                    "Which dot is closest to the target? Answer D<number>:",
                    api_key, model, max_tokens=10)
    if not text:
        return None
    m = re.match(r"D(\d+)", text.upper())
    if m:
        num = int(m.group(1))
        return num if 1 <= num <= num_dots else None
    return None


def pinpoint_ask_majority(b64_image: str, anchor: str, num_dots: int,
                           api_key: str, model: str, votes: int = 3,
                           query_counter: list | None = None) -> tuple[int | None, bool]:
    """Ask VLM which dot is closest, with majority vote for robustness."""
    from collections import Counter
    results = []
    for _ in range(votes):
        if query_counter is not None:
            query_counter[0] += 1
        hit = pinpoint_ask_closest(b64_image, anchor, num_dots, api_key, model)
        results.append(hit)
    counter = Counter(r for r in results if r is not None)
    if not counter:
        return None, False
    best, count = counter.most_common(1)[0]
    if count >= (votes + 1) // 2:
        return best, count == votes
    return None, False
