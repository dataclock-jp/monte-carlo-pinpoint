"""
Template Matcher — OpenCV テンプレートマッチングによる汎用UI要素検索

アプリの種類を問わず、画面上のUI要素を画像パターンで検索する。
UI Automation が効かないアプリ（ゲーム、レガシーアプリ、独自描画）でも動作する。

テンプレート画像は app_knowledge/templates/{app_name}/ に保存される。
app_knowledge を作成する際にスクリーンショットやWeb検索で得た画像を
テンプレートとして登録すれば、以降は ~50ms で座標を取得できる。

使い方:
    matcher = TemplateMatcher()

    # テンプレート保存（スクリーンショットの領域を切り出して保存）
    matcher.save_template("notepad", "save_button", screenshot, x=100, y=50, w=30, h=30)

    # テンプレート検索（画面上から一致する位置を検索）
    results = matcher.find(screenshot, "notepad", "save_button")
    # → [Match(center=(115, 65), score=0.95, rect=(100, 50, 30, 30))]

    # 全テンプレートから検索
    results = matcher.find_any(screenshot, "notepad", threshold=0.8)

    # テンプレート一覧
    templates = matcher.list_templates("notepad")
"""

import os
import logging
import json
import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# テンプレート保存先
_DEFAULT_TEMPLATE_ROOT = Path("D:/ClaudeProject/app_knowledge/templates")


@dataclass
class Match:
    """テンプレートマッチングの結果"""
    template_name: str
    center_x: int
    center_y: int
    score: float  # 0.0-1.0 (1.0 = 完全一致)
    rect: tuple[int, int, int, int]  # (x, y, w, h)
    app: str = ""

    def __str__(self):
        return (f"[{self.app}/{self.template_name}] "
                f"center=({self.center_x},{self.center_y}) "
                f"score={self.score:.3f} "
                f"rect=({self.rect[0]},{self.rect[1]},{self.rect[2]}x{self.rect[3]})")


class TemplateMatcher:
    """OpenCV テンプレートマッチングによるUI要素検索"""

    def __init__(self, template_root: str | Path = ""):
        self.template_root = Path(template_root) if template_root else _DEFAULT_TEMPLATE_ROOT
        self.template_root.mkdir(parents=True, exist_ok=True)
        # テンプレートキャッシュ（読み込み済み画像）
        self._cache: dict[str, np.ndarray] = {}

    def _app_dir(self, app: str) -> Path:
        """アプリ別テンプレートディレクトリ"""
        safe_name = app.replace(" ", "_").replace("/", "_").replace("\\", "_")
        d = self.template_root / safe_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _meta_path(self, app: str) -> Path:
        """アプリのテンプレートメタデータファイル"""
        return self._app_dir(app) / "_meta.json"

    def _load_meta(self, app: str) -> dict:
        mp = self._meta_path(app)
        if mp.exists():
            try:
                return json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_meta(self, app: str, meta: dict):
        mp = self._meta_path(app)
        mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # === テンプレート保存 ===

    def save_template(
        self,
        app: str,
        name: str,
        image: np.ndarray,
        x: int, y: int, w: int, h: int,
        description: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """スクリーンショットから領域を切り出してテンプレートとして保存する。

        Args:
            app: アプリ名（例: "notepad", "clip_studio_paint"）
            name: テンプレート名（例: "save_button", "color_wheel"）
            image: スクリーンショット画像（BGR numpy array）
            x, y, w, h: 切り出し領域（画像内座標）
            description: テンプレートの説明
            tags: タグ（例: ["toolbar", "button"]）

        Returns:
            保存先パスの文字列
        """
        # 領域を切り出し
        crop = image[y:y+h, x:x+w].copy()
        if crop.size == 0:
            return "エラー: 領域が空です"

        # 保存（cv2.imwrite は非ASCII パスに未対応なので imencode + write）
        safe_name = name.replace(" ", "_")
        path = self._app_dir(app) / f"{safe_name}.png"
        _, buf = cv2.imencode(".png", crop)
        path.write_bytes(buf.tobytes())

        # メタデータ更新
        meta = self._load_meta(app)
        meta[safe_name] = {
            "description": description,
            "tags": tags or [],
            "size": [w, h],
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save_meta(app, meta)

        # キャッシュ更新
        cache_key = f"{app}/{safe_name}"
        self._cache[cache_key] = crop

        logger.info(f"テンプレート保存: {path} ({w}x{h})")
        return str(path)

    def save_template_from_file(
        self,
        app: str,
        name: str,
        image_path: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """既存の画像ファイルをテンプレートとして登録する。

        app_knowledge 作成時にWeb検索で得た画像を登録する場合に使う。
        """
        # cv2.imread も非ASCII パス未対応なので numpy 経由で読む
        img_path = Path(image_path)
        if not img_path.exists():
            return f"エラー: ファイルが見つかりません: {image_path}"
        buf = np.frombuffer(img_path.read_bytes(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return f"エラー: 画像を読み込めません: {image_path}"

        safe_name = name.replace(" ", "_")
        path = self._app_dir(app) / f"{safe_name}.png"
        _, enc = cv2.imencode(".png", img)
        path.write_bytes(enc.tobytes())

        meta = self._load_meta(app)
        h, w = img.shape[:2]
        meta[safe_name] = {
            "description": description,
            "tags": tags or [],
            "size": [w, h],
            "source": image_path,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save_meta(app, meta)

        cache_key = f"{app}/{safe_name}"
        self._cache[cache_key] = img

        return str(path)

    # === テンプレート読み込み ===

    def _load_template(self, app: str, name: str) -> Optional[np.ndarray]:
        """テンプレート画像を読み込む（キャッシュ付き）"""
        cache_key = f"{app}/{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self._app_dir(app) / f"{name}.png"
        if not path.exists():
            return None

        buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            self._cache[cache_key] = img
        return img

    # === テンプレートマッチング ===

    def find(
        self,
        screenshot: np.ndarray,
        app: str,
        name: str,
        threshold: float = 0.8,
        max_results: int = 5,
        scales: list[float] | None = None,
    ) -> list[Match]:
        """指定テンプレートを画面から検索する。

        Args:
            screenshot: 検索対象の画像（BGR numpy array）
            app: アプリ名
            name: テンプレート名
            threshold: 一致度の閾値（0-1）
            max_results: 最大結果数
            scales: マルチスケール検索の倍率リスト。None=[1.0]

        Returns:
            Match のリスト（score 降順）
        """
        template = self._load_template(app, name)
        if template is None:
            return []

        if scales is None:
            scales = [1.0]

        all_matches = []

        for scale in scales:
            if scale != 1.0:
                th, tw = template.shape[:2]
                new_w = max(1, int(tw * scale))
                new_h = max(1, int(th * scale))
                tmpl = cv2.resize(template, (new_w, new_h))
            else:
                tmpl = template

            th, tw = tmpl.shape[:2]
            sh, sw = screenshot.shape[:2]
            if tw > sw or th > sh:
                continue

            # テンプレートマッチング
            result = cv2.matchTemplate(screenshot, tmpl, cv2.TM_CCOEFF_NORMED)

            # 閾値以上の箇所を検出
            locations = np.where(result >= threshold)
            for pt_y, pt_x in zip(*locations):
                score = float(result[pt_y, pt_x])
                all_matches.append(Match(
                    template_name=name,
                    center_x=pt_x + tw // 2,
                    center_y=pt_y + th // 2,
                    score=score,
                    rect=(int(pt_x), int(pt_y), tw, th),
                    app=app,
                ))

        # 重複排除（近接マッチを統合）
        all_matches.sort(key=lambda m: m.score, reverse=True)
        filtered = []
        for m in all_matches:
            if len(filtered) >= max_results:
                break
            # 既存結果と近すぎるものは除外
            too_close = False
            for f in filtered:
                if abs(m.center_x - f.center_x) < 10 and abs(m.center_y - f.center_y) < 10:
                    too_close = True
                    break
            if not too_close:
                filtered.append(m)

        return filtered

    def find_any(
        self,
        screenshot: np.ndarray,
        app: str,
        threshold: float = 0.8,
        tags: list[str] | None = None,
        scales: list[float] | None = None,
    ) -> list[Match]:
        """アプリの全テンプレート（またはタグ指定）から検索する。

        Args:
            screenshot: 検索対象の画像
            app: アプリ名
            threshold: 一致度の閾値
            tags: タグでフィルタ（指定したタグを持つテンプレートのみ）
            scales: マルチスケール検索の倍率リスト
        """
        meta = self._load_meta(app)
        results = []

        for name, info in meta.items():
            if tags:
                tmpl_tags = info.get("tags", [])
                if not any(t in tmpl_tags for t in tags):
                    continue
            matches = self.find(screenshot, app, name,
                                threshold=threshold, scales=scales)
            results.extend(matches)

        results.sort(key=lambda m: m.score, reverse=True)
        return results

    # === マルチスケール検索 ===

    def find_multiscale(
        self,
        screenshot: np.ndarray,
        app: str,
        name: str,
        threshold: float = 0.75,
        scale_range: tuple[float, float] = (0.3, 2.0),
        scale_steps: int = 7,
    ) -> list[Match]:
        """マルチスケールでテンプレート検索（DPIやリサイズ対応）。

        テンプレートのサイズが画面上のサイズと異なる場合に使う。
        """
        lo, hi = scale_range
        scales = [lo + (hi - lo) * i / max(1, scale_steps - 1)
                  for i in range(scale_steps)]
        return self.find(screenshot, app, name,
                         threshold=threshold, scales=scales)

    # === テンプレート管理 ===

    def list_templates(self, app: str = "") -> dict[str, list[dict]]:
        """テンプレート一覧を返す。

        Args:
            app: アプリ名（空で全アプリ）

        Returns:
            {app_name: [{name, description, tags, size}, ...]}
        """
        result = {}
        if app:
            apps = [app]
        else:
            apps = [d.name for d in self.template_root.iterdir() if d.is_dir()]

        for a in apps:
            meta = self._load_meta(a)
            if meta:
                templates = []
                for name, info in meta.items():
                    templates.append({
                        "name": name,
                        "description": info.get("description", ""),
                        "tags": info.get("tags", []),
                        "size": info.get("size", [0, 0]),
                    })
                result[a] = templates

        return result

    def delete_template(self, app: str, name: str) -> bool:
        """テンプレートを削除"""
        path = self._app_dir(app) / f"{name}.png"
        if path.exists():
            path.unlink()
        cache_key = f"{app}/{name}"
        self._cache.pop(cache_key, None)
        meta = self._load_meta(app)
        if name in meta:
            del meta[name]
            self._save_meta(app, meta)
            return True
        return False

    def clear_cache(self):
        """キャッシュをクリア"""
        self._cache.clear()


# ===========================================================================
# LLM ベースのテンプレート学習（自動セル選択で領域を絞り込み、テンプレ保存）
# ===========================================================================

def _draw_grid_on_image(img: np.ndarray, grid_size: int = 5) -> np.ndarray:
    """画像にグリッドとセルラベルを描画する（LLM に送る用）。"""
    out = img.copy()
    h, w = out.shape[:2]
    cell_w = w // grid_size
    cell_h = h // grid_size

    # グリッド線（緑）
    for i in range(1, grid_size):
        cv2.line(out, (cell_w * i, 0), (cell_w * i, h), (0, 255, 0), 2)
        cv2.line(out, (0, cell_h * i), (w, cell_h * i), (0, 255, 0), 2)
    # 外枠
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 255, 0), 2)

    # セルラベル
    cols = "ABCDEFGHIJ"[:grid_size]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(cell_w, cell_h) / 150)
    for r in range(grid_size):
        for c in range(grid_size):
            label = f"{cols[c]}{r + 1}"
            lx = c * cell_w + 4
            ly = r * cell_h + int(20 * font_scale) + 4
            # 背景
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
            cv2.rectangle(out, (lx - 2, ly - th - 4), (lx + tw + 2, ly + 2),
                          (0, 0, 0), -1)
            cv2.putText(out, label, (lx, ly), font, font_scale,
                        (0, 255, 0), 1, cv2.LINE_AA)
    return out


def _image_to_base64_jpeg(img: np.ndarray, quality: int = 80) -> str:
    """BGR numpy array → base64 JPEG 文字列"""
    import base64
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("ascii")


def verify_match_with_llm(
    screenshot: np.ndarray,
    match: "Match",
    target_name: str,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    padding: int = 40,
) -> bool:
    """低信頼テンプレートマッチの結果を LLM で検証する。

    マッチ領域を切り出し、赤枠で囲んだ画像を LLM に見せて
    「これは {target_name} ですか？」と確認する。

    Args:
        screenshot: 元のスクリーンショット (BGR)
        match: テンプレートマッチの結果
        target_name: 探しているUI要素の名前
        api_key: Anthropic API キー
        model: 使用する Claude モデル
        padding: マッチ領域の周辺余白 (px)

    Returns:
        True = LLM が「これは target_name」と判定, False = 違う
    """
    import requests as _req

    sh, sw = screenshot.shape[:2]
    rx, ry, rw, rh = match.rect

    # padding 付きで切り出し
    x1 = max(0, rx - padding)
    y1 = max(0, ry - padding)
    x2 = min(sw, rx + rw + padding)
    y2 = min(sh, ry + rh + padding)
    crop = screenshot[y1:y2, x1:x2].copy()

    # マッチ領域を赤枠で囲む
    cv2.rectangle(crop,
                  (rx - x1, ry - y1),
                  (rx - x1 + rw, ry - y1 + rh),
                  (0, 0, 255), 2)

    # 小さすぎる場合は拡大
    ch, cw = crop.shape[:2]
    if max(cw, ch) < 128:
        scale = 128 / max(cw, ch)
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                          interpolation=cv2.INTER_CUBIC)

    b64 = _image_to_base64_jpeg(crop, quality=85)

    system = (
        "You verify whether a UI element matches a given name. "
        "The image shows a region with a red rectangle marking the candidate. "
        "Answer YES if the marked element is the target, NO otherwise. "
        "Output only YES or NO."
    )
    user = f"Is the element inside the red rectangle \"{target_name}\"?"

    payload = {
        "model": model,
        "max_tokens": 4,
        "system": system,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                },
                {"type": "text", "text": user},
            ],
        }],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        resp = _req.post(
            "https://api.anthropic.com/v1/messages",
            json=payload, headers=headers, timeout=30,
        )
        if resp.status_code != 200:
            logger.warning("verify_match LLM error: %s", resp.status_code)
            return False
        text = resp.json()["content"][0]["text"].strip().upper()
        return "YES" in text
    except Exception as e:
        logger.warning("verify_match LLM failed: %s", e)
        return False


def _ask_llm_cell(
    b64_image: str,
    target_description: str,
    grid_size: int,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
) -> str | None:
    """LLM に画像を見せて、ターゲットが含まれるセル名を聞く。

    Returns:
        セル名（例: "C2"）。見つからなければ None。
    """
    import requests as _req

    cols = "ABCDEFGHIJ"[:grid_size]
    valid_cells = [f"{c}{r+1}" for c in cols for r in range(grid_size)]

    system = (
        "あなたは画面上のUI要素を特定するアシスタントです。"
        "画像にはグリッド線とセルラベル（A1, B2等）が描かれています。"
        "指定されたターゲットが含まれるセルを1つだけ答えてください。"
        "セル名のみを回答してください（例: C2）。余計な説明は不要です。"
        "ターゲットが画像内に見つからない場合は NONE と答えてください。"
    )
    user = (
        f"ターゲット: {target_description}\n"
        f"有効なセル名: {', '.join(valid_cells)}\n"
        f"ターゲットが含まれるセルを1つだけ答えてください。"
    )

    payload = {
        "model": model,
        "max_tokens": 16,
        "system": system,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64_image,
                    },
                },
                {"type": "text", "text": user},
            ],
        }],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        resp = _req.post(
            "https://api.anthropic.com/v1/messages",
            json=payload, headers=headers, timeout=30,
        )
        if resp.status_code != 200:
            logger.warning(f"LLM API error: {resp.status_code} {resp.text[:200]}")
            return None
        text = resp.json()["content"][0]["text"].strip().upper()
        # セル名を抽出（"C2" や "C2です" から "C2" を取る）
        for cell in valid_cells:
            if cell in text:
                return cell
        if "NONE" in text:
            return None
        logger.warning(f"LLM 応答からセル名を抽出できません: {text}")
        return None
    except Exception as e:
        logger.warning(f"LLM API 呼び出し失敗: {e}")
        return None


def _check_image_quality(crop: np.ndarray) -> tuple[bool, str]:
    """テンプレート画像の品質を事前チェックする（LLM不要）。

    コントラストやエントロピーが低すぎる画像（ほぼ単色の背景等）を弾く。

    Returns:
        (passed, reason): passed=True なら OK、False なら不合格（reason に理由）
    """
    if crop.size == 0:
        return False, "画像が空です"

    h, w = crop.shape[:2]
    if h < 4 or w < 4:
        return False, f"画像が小さすぎます ({w}x{h}px)"

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

    # コントラスト（標準偏差）: 低すぎればほぼ単色
    std = float(np.std(gray))
    if std < 8.0:
        return False, f"コントラストが低すぎます (std={std:.1f}, 閾値=8.0)。ほぼ単色の背景です"

    # エントロピー: 情報量が少なすぎれば模様のない背景
    # ※ ダークテーマのUIアイコンは色数が少なくエントロピーが低いため、閾値は緩め
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist[hist > 0] / hist.sum()
    entropy = float(-np.sum(hist * np.log2(hist)))
    if entropy < 1.0:
        return False, f"エントロピーが低すぎます ({entropy:.2f}, 閾値=1.0)。情報量が不足しています"

    # エッジ密度: UI要素にはエッジ（輪郭）があるはず
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.count_nonzero(edges)) / (h * w)
    if edge_ratio < 0.005:
        return False, f"エッジが少なすぎます ({edge_ratio:.4f}, 閾値=0.005)。明確な形状がありません"

    return True, "OK"


def _verify_template(
    crop: np.ndarray,
    target_description: str,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
) -> bool:
    """切り出したテンプレート画像が本当にターゲットかどうかを検証する。

    Step 1: 画像品質チェック（コントラスト、エントロピー、エッジ密度）
    Step 2: LLM 視覚検証（拡大画像で厳格に判定）

    Returns:
        True: ターゲットと一致、False: 不一致（背景や無関係な画像）
    """
    import requests as _req

    # Step 1: 画像品質の事前チェック
    passed, reason = _check_image_quality(crop)
    if not passed:
        logger.warning(f"テンプレート品質チェック不合格: {reason}")
        return False

    # Step 2: LLM 検証用に画像を拡大（小さい画像はLLMが判断しづらい）
    h, w = crop.shape[:2]
    min_dim = 128
    if w < min_dim or h < min_dim:
        scale = max(min_dim / w, min_dim / h)
        crop_resized = cv2.resize(crop, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_NEAREST)
    else:
        crop_resized = crop

    b64 = _image_to_base64_jpeg(crop_resized)

    system = (
        "あなたはUI要素のテンプレート画像を厳格に検証する品質管理者です。"
        "提示された画像が、指定されたターゲットUI要素のテンプレートとして使えるかを判定します。"
        "必ず YES または NO のみで回答してください。"
    )
    user = (
        f"この画像は「{target_description}」のテンプレート画像として適切ですか？\n\n"
        f"以下の場合は必ず NO と回答してください:\n"
        f"- ほぼ単色の背景（黒、灰色、白など）しか写っていない\n"
        f"- ターゲットのUI要素（ボタン、アイコン、ラベル）が明確に識別できない\n"
        f"- 背景の一部、壁紙、グラデーションのみ\n"
        f"- 別のUI要素や無関係なテキストが主に写っている\n"
        f"- ぼやけていて何の要素か判別できない\n\n"
        f"YES の条件: ターゲット「{target_description}」が画像の中心付近に明確に写っていること。\n\n"
        f"YESまたはNOのみで回答してください。"
    )

    payload = {
        "model": model,
        "max_tokens": 8,
        "system": system,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                },
                {"type": "text", "text": user},
            ],
        }],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        resp = _req.post(
            "https://api.anthropic.com/v1/messages",
            json=payload, headers=headers, timeout=30,
        )
        if resp.status_code != 200:
            logger.warning(f"テンプレート検証 API error: {resp.status_code}")
            return False  # API エラー時は保存しない（安全側に倒す）
        text = resp.json()["content"][0]["text"].strip().upper()
        if "YES" not in text:
            logger.warning(f"テンプレート検証失敗: LLM が '{target_description}' と判定しませんでした (応答: {text})")
            return False
        return True
    except Exception as e:
        logger.warning(f"テンプレート検証例外: {e}")
        return False  # 例外時も保存しない（安全側に倒す）


def learn_template(
    screenshot: np.ndarray,
    target_description: str,
    app: str,
    name: str,
    api_key: str,
    matcher: TemplateMatcher,
    grid_size: int = 5,
    max_steps: int = 4,
    target_px: int = 60,
    model: str = "claude-sonnet-4-20250514",
    monitor_phys_w: int = 3840,
    screenshot_w: int = 1920,
) -> dict:
    """LLM の視覚を使って画面からターゲットを段階的に絞り込み、テンプレートとして保存する。

    Args:
        screenshot: 画面全体の BGR 画像（numpy array）
        target_description: 探すUI要素の説明（例: "歯車アイコンの設定ボタン"）
        app: アプリ名（テンプレート保存先）
        name: テンプレート名
        api_key: Anthropic API キー
        matcher: TemplateMatcher インスタンス
        grid_size: グリッド分割数（5推奨）
        max_steps: 最大ステップ数
        target_px: 目標セルサイズ（物理ピクセル）。これ以下で停止
        model: 使用する Claude モデル
        monitor_phys_w: モニターの物理幅（物理セルサイズ計算用）
        screenshot_w: スクリーンショットの幅

    Returns:
        {
            "success": bool,
            "path": str,         # 保存先パス
            "steps": int,        # 実行ステップ数
            "history": [str],    # セル選択履歴
            "elapsed": float,    # 合計秒数
            "region": (x,y,w,h), # 最終領域（スクリーンショット座標）
            "error": str,        # エラー時のメッセージ
        }
    """
    t0 = time.time()
    history = []
    rx, ry = 0, 0
    rw, rh = screenshot.shape[1], screenshot.shape[0]
    phys_ratio = monitor_phys_w / screenshot_w

    for step in range(max_steps):
        # 物理セルサイズをチェック
        phys_cell_w = (rw / grid_size) * phys_ratio
        if phys_cell_w <= target_px and step > 0:
            break

        # 現在の領域を切り出し
        crop = screenshot[ry:ry+rh, rx:rx+rw]
        if crop.size == 0:
            return {"success": False, "error": "領域が空", "steps": step,
                    "history": history, "elapsed": time.time() - t0,
                    "region": (rx, ry, rw, rh), "path": ""}

        # グリッド描画 → base64
        gridded = _draw_grid_on_image(crop, grid_size)
        b64 = _image_to_base64_jpeg(gridded)

        # LLM に質問
        cell = _ask_llm_cell(b64, target_description, grid_size, api_key, model)
        if cell is None:
            return {"success": False, "error": f"ステップ{step+1}: LLMがターゲットを見つけられませんでした",
                    "steps": step + 1, "history": history,
                    "elapsed": time.time() - t0,
                    "region": (rx, ry, rw, rh), "path": ""}

        history.append(cell)

        # セルの領域を計算
        col = ord(cell[0]) - ord("A")
        row = int(cell[1]) - 1
        cell_w = rw // grid_size
        cell_h = rh // grid_size
        rx = rx + col * cell_w
        ry = ry + row * cell_h
        rw = cell_w
        rh = cell_h

    # === 保存前検証: 切り出した画像が本当にターゲットか LLM に確認 ===
    crop_for_verify = screenshot[ry:ry+rh, rx:rx+rw]
    if crop_for_verify.size > 0:
        verified = _verify_template(crop_for_verify, target_description, api_key, model)
        if not verified:
            elapsed = time.time() - t0
            return {
                "success": False,
                "error": f"検証失敗: 切り出した画像はターゲット（{target_description}）と一致しませんでした。"
                         f"背景や無関係な領域を誤検出した可能性があります。",
                "steps": len(history),
                "history": history,
                "elapsed": elapsed,
                "region": (rx, ry, rw, rh),
                "path": "",
            }

    # テンプレートとして保存
    path = matcher.save_template(
        app, name, screenshot, x=rx, y=ry, w=rw, h=rh,
        description=target_description,
        tags=["auto_learned"],
    )

    elapsed = time.time() - t0
    return {
        "success": True,
        "path": path,
        "steps": len(history),
        "history": history,
        "elapsed": elapsed,
        "region": (rx, ry, rw, rh),
        "error": "",
    }


# ============================================================================
# UIA 座標ガイド学習（方法1: UIA 対応アプリ向け）
# ============================================================================

def learn_template_from_uia(
    element_center: tuple[int, int],
    element_size: tuple[int, int],
    target_description: str,
    app: str,
    name: str,
    api_key: str,
    matcher: TemplateMatcher,
    padding: int = 8,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """UI Automation で取得した座標からテンプレートを保存する。

    watcher_find で得た物理座標・サイズを元に、画面からアイコンを
    ピクセル精度で切り出す。グリッドズーム不要で、20px 未満の
    小さなアイコンも確実にキャプチャできる。

    Args:
        element_center: 要素の中心座標 (物理ピクセル)
        element_size: 要素のサイズ (w, h) (物理ピクセル)
        target_description: 要素の説明
        app: アプリ名
        name: テンプレート名
        api_key: Anthropic API キー
        matcher: TemplateMatcher インスタンス
        padding: 要素周囲の余白ピクセル
        model: 検証用 Claude モデル

    Returns:
        {"success": bool, "path": str, "elapsed": float, "size": (w,h), "error": str}
    """
    import mss

    t0 = time.time()
    cx, cy = element_center
    ew, eh = element_size

    # 切り出し領域を計算（padding 付き）
    left = cx - ew // 2 - padding
    top = cy - eh // 2 - padding
    cap_w = ew + padding * 2
    cap_h = eh + padding * 2

    # mss でフル解像度キャプチャ（リサイズなし）
    try:
        with mss.mss() as sct:
            region = {"left": left, "top": top, "width": cap_w, "height": cap_h}
            grab = sct.grab(region)
            crop = np.array(grab)[:, :, :3].copy()  # BGRA → BGR（OpenCV 準拠）
    except Exception as e:
        return {"success": False, "path": "", "elapsed": time.time() - t0,
                "size": (0, 0), "error": f"キャプチャ失敗: {e}"}

    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return {"success": False, "path": "", "elapsed": time.time() - t0,
                "size": (crop.shape[1], crop.shape[0]),
                "error": "切り出し画像が小さすぎます"}

    # 品質チェック + LLM 検証
    verified = _verify_template(crop, target_description, api_key, model)
    if not verified:
        elapsed = time.time() - t0
        return {
            "success": False, "path": "", "elapsed": elapsed,
            "size": (crop.shape[1], crop.shape[0]),
            "error": f"検証失敗: 切り出し画像は「{target_description}」と一致しませんでした",
        }

    # テンプレートとして保存（切り出し済み画像をそのまま保存）
    safe_app = app.replace("/", "_").replace("\\", "_")
    tmpl_dir = matcher.template_root / safe_app
    tmpl_dir.mkdir(parents=True, exist_ok=True)

    save_path = tmpl_dir / f"{name}.png"
    ok, encoded = cv2.imencode(".png", crop)
    if ok:
        save_path.write_bytes(encoded.tobytes())
    else:
        return {"success": False, "path": "", "elapsed": time.time() - t0,
                "size": (crop.shape[1], crop.shape[0]),
                "error": "画像エンコード失敗"}

    # メタデータ更新
    meta = matcher._load_meta(safe_app)
    meta[name] = {
        "description": target_description,
        "tags": ["uia_learned"],
        "size": [crop.shape[1], crop.shape[0]],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    matcher._save_meta(safe_app, meta)

    elapsed = time.time() - t0
    return {
        "success": True,
        "path": str(save_path),
        "elapsed": elapsed,
        "size": (crop.shape[1], crop.shape[0]),
        "error": "",
    }


# ============================================================================
# ツールバー自動分割学習（方法2: UIA 非対応アプリ向け）
# ============================================================================

def detect_ui_elements(
    screenshot: np.ndarray,
    min_size: int = 15,
    max_size: int = 300,
    max_aspect: float = 6.0,
) -> list[dict]:
    """画面上のUI要素を輪郭検出する（SoM用）。

    _segment_toolbar_icons を汎用化したもの。画面全体やウィンドウ全体で動作。

    Args:
        screenshot: BGR 画像
        min_size: 最小要素サイズ (px)
        max_size: 最大要素サイズ (px)
        max_aspect: 最大アスペクト比

    Returns:
        [{"rect": (x, y, w, h), "center": (cx, cy)}, ...]
    """
    h, w = screenshot.shape[:2]
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # 方式1: Canny エッジ検出（ダークテーマ・ライトテーマ両対応）
    edges = cv2.Canny(gray, 30, 100)

    # 方式2: 背景色差分（_segment_toolbar_icons と同じ手法）
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    bg_val = int(np.argmax(hist))
    diff = np.abs(gray.astype(np.int16) - bg_val).astype(np.uint8)
    _, bg_binary = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

    # 両方式を OR 結合
    binary = cv2.bitwise_or(edges, bg_binary)

    # モルフォロジーで近いピクセルを結合
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, kernel, iterations=2)
    binary = cv2.erode(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    elements = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < min_size or ch < min_size:
            continue
        if cw > max_size or ch > max_size:
            continue
        aspect = max(cw, ch) / max(min(cw, ch), 1)
        if aspect > max_aspect:
            continue

        elements.append({
            "rect": (x, y, cw, ch),
            "center": (x + cw // 2, y + ch // 2),
        })

    # 左上から右下にソート
    elements.sort(key=lambda e: (e["rect"][1] // 30, e["rect"][0]))
    return elements


def _segment_toolbar_icons(
    toolbar_crop: np.ndarray,
    min_icon_px: int = 10,
    max_icon_px: int = 80,
) -> list[dict]:
    """ツールバー画像からアイコンを個別に分割する。

    エッジ検出 + 輪郭検出で個々のアイコン領域を抽出。

    Args:
        toolbar_crop: ツールバー領域の BGR 画像
        min_icon_px: 最小アイコンサイズ
        max_icon_px: 最大アイコンサイズ

    Returns:
        [{"image": np.ndarray, "rect": (x, y, w, h)}, ...]
    """
    h, w = toolbar_crop.shape[:2]
    gray = cv2.cvtColor(toolbar_crop, cv2.COLOR_BGR2GRAY)

    # 方法A: エッジ+輪郭ベース
    # 背景色を推定（最頻値）して二値化
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    bg_val = int(np.argmax(hist))

    # 背景との差が大きいピクセルをマスク
    diff = np.abs(gray.astype(np.int16) - bg_val).astype(np.uint8)
    _, binary = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

    # モルフォロジーで近いピクセルを結合
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, kernel, iterations=2)
    binary = cv2.erode(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    icons = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        # サイズフィルタ
        if cw < min_icon_px or ch < min_icon_px:
            continue
        if cw > max_icon_px or ch > max_icon_px:
            continue
        # 正方形に近い or 細長すぎないこと
        aspect = max(cw, ch) / max(min(cw, ch), 1)
        if aspect > 4.0:
            continue

        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + cw + pad)
        y1 = min(h, y + ch + pad)
        icon_img = toolbar_crop[y0:y1, x0:x1]
        if icon_img.size > 0:
            icons.append({
                "image": icon_img,
                "rect": (x0, y0, x1 - x0, y1 - y0),
            })

    # 左上から右下にソート（ツールバーの並び順）
    icons.sort(key=lambda ic: (ic["rect"][1] // 20, ic["rect"][0]))
    return icons


def _identify_icon(
    icon_img: np.ndarray,
    api_key: str,
    app_name: str = "",
    app_knowledge: str = "",
    model: str = "claude-sonnet-4-20250514",
) -> str | None:
    """アイコン画像をLLMに見せて、何のツール/ボタンかを同定する。

    Args:
        icon_img: アイコン画像 (BGR numpy array)
        api_key: Anthropic API キー
        app_name: アプリ名
        app_knowledge: app_knowledge のテキスト（UI要素名・ツール名の参考情報）
        model: Claude モデル

    Returns:
        (name, description) タプル or None（判別不能時）
    """
    import requests as _req

    # 小さい画像を拡大
    h, w = icon_img.shape[:2]
    min_dim = 128
    if w < min_dim or h < min_dim:
        scale = max(min_dim / w, min_dim / h)
        icon_resized = cv2.resize(icon_img, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_NEAREST)
    else:
        icon_resized = icon_img

    b64 = _image_to_base64_jpeg(icon_resized)

    # app_knowledge からUI要素名を抽出して参考情報にする
    knowledge_hint = ""
    if app_knowledge:
        # 最大500文字に制限（プロンプト肥大化防止）
        knowledge_hint = (
            f"\n\nこのアプリの既知のUI要素・ツール名:\n"
            f"{app_knowledge[:500]}\n"
            f"上記の名前に合致するものがあれば、その名前を使ってください。\n"
        )

    system = "あなたはUIアイコンを識別するアシスタントです。短く正確に回答してください。"
    user = (
        f"この画像に写っているUI要素は何ですか？\n"
        f"{'アプリ: ' + app_name + chr(10) if app_name else ''}"
        f"回答形式: 1行目に英語スネークケースのID（例: pen_tool, eraser, save_button）、"
        f"2行目に日本語の説明（例: ペンツールのアイコン）。\n"
        f"判別できない場合は UNKNOWN とだけ回答してください。"
        f"{knowledge_hint}"
    )

    payload = {
        "model": model,
        "max_tokens": 50,
        "system": system,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/jpeg",
                                             "data": b64}},
                {"type": "text", "text": user},
            ],
        }],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        resp = _req.post("https://api.anthropic.com/v1/messages",
                         json=payload, headers=headers, timeout=30)
        if resp.status_code != 200:
            return None
        text = resp.json()["content"][0]["text"].strip()
        if "UNKNOWN" in text.upper():
            return None
        lines = text.strip().split("\n")
        name = lines[0].strip().lower().replace(" ", "_").replace("-", "_")
        # 不正な文字を除去
        name = re.sub(r'[^a-z0-9_]', '', name)
        description = lines[1].strip() if len(lines) > 1 else ""
        logger.info(f"アイコン同定: {name} — {description}")
        return name, description
    except Exception as e:
        logger.warning(f"アイコン同定失敗: {e}")
        return None


def learn_template_toolbar(
    screenshot: np.ndarray,
    toolbar_rect: tuple[int, int, int, int],
    app: str,
    api_key: str,
    matcher: TemplateMatcher,
    max_icons: int = 20,
    app_knowledge: str = "",
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """ツールバー領域からアイコンを自動分割して一括学習する。

    UIA 非対応アプリ向け。フル解像度のスクリーンショットから
    ツールバー領域を切り出し、OpenCV でアイコンを個別分割し、
    LLM で各アイコンを同定してテンプレートとして保存する。

    Args:
        screenshot: フル解像度の BGR 画像（リサイズなし）
        toolbar_rect: ツールバー領域 (x, y, w, h) ピクセル座標
        app: アプリ名
        api_key: Anthropic API キー
        matcher: TemplateMatcher インスタンス
        max_icons: 最大学習アイコン数
        model: 使用する Claude モデル

    Returns:
        {"success": bool, "learned": [str], "failed": [str],
         "total_icons": int, "elapsed": float}
    """
    t0 = time.time()
    tx, ty, tw, th = toolbar_rect

    # ツールバー領域を切り出し
    toolbar_crop = screenshot[ty:ty+th, tx:tx+tw]
    if toolbar_crop.size == 0:
        return {"success": False, "learned": [], "failed": [],
                "total_icons": 0, "elapsed": time.time() - t0,
                "error": "ツールバー領域が空です"}

    # アイコンを分割
    icons = _segment_toolbar_icons(toolbar_crop)
    if not icons:
        return {"success": False, "learned": [], "failed": [],
                "total_icons": 0, "elapsed": time.time() - t0,
                "error": "アイコンが検出されませんでした"}

    learned = []
    failed = []

    for icon_data in icons[:max_icons]:
        icon_img = icon_data["image"]

        # LLM でアイコンを同定
        id_result = _identify_icon(icon_img, api_key, app, app_knowledge, model)
        if id_result is None:
            failed.append("unknown")
            continue

        icon_name, description = id_result

        if not icon_name or icon_name == "unknown":
            failed.append("unknown")
            continue

        # 既存テンプレートをスキップ
        existing = matcher.list_templates(app)
        existing_names = [t["name"] for t in existing.get(app, [])]
        if icon_name in existing_names:
            learned.append(f"{icon_name} (既存スキップ)")
            continue

        # 品質チェック + LLM 検証
        verified = _verify_template(icon_img, description or icon_name,
                                    api_key, model)
        if not verified:
            failed.append(f"{icon_name} (検証失敗)")
            continue

        # 保存
        safe_app = app.replace("/", "_").replace("\\", "_")
        tmpl_dir = matcher.template_root / safe_app
        tmpl_dir.mkdir(parents=True, exist_ok=True)

        save_path = tmpl_dir / f"{icon_name}.png"
        ok, encoded = cv2.imencode(".png", icon_img)
        if ok:
            save_path.write_bytes(encoded.tobytes())
            meta = matcher._load_meta(safe_app)
            meta[icon_name] = {
                "description": description or icon_name,
                "tags": ["toolbar_learned"],
                "size": [icon_img.shape[1], icon_img.shape[0]],
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            matcher._save_meta(safe_app, meta)
            learned.append(icon_name)
        else:
            failed.append(f"{icon_name} (エンコード失敗)")

    elapsed = time.time() - t0
    return {
        "success": len(learned) > 0,
        "learned": learned,
        "failed": failed,
        "total_icons": len(icons),
        "elapsed": elapsed,
        "error": "",
    }


# ============================================================================
# Auto-learn: 画面から主要 UI 要素を一括学習
# ============================================================================

def _identify_ui_elements(
    b64_screenshot: str,
    app_name: str,
    api_key: str,
    app_knowledge: str = "",
    max_elements: int = 10,
    model: str = "claude-sonnet-4-20250514",
) -> list[dict]:
    """スクリーンショットから主要 UI 要素を特定する。

    Returns:
        [{"name": "pen_tool", "description": "ペンツールアイコン（左ツールバー）"}, ...]
    """
    knowledge_section = ""
    if app_knowledge:
        truncated = app_knowledge[:3000]
        knowledge_section = (
            f"\n\n以下はこのアプリの既知情報です。要素名の参考にしてください:\n"
            f"---\n{truncated}\n---"
        )

    system = (
        f"あなたは {app_name} の UI 分析エキスパートです。\n"
        f"スクリーンショットを見て、テンプレートマッチングで検索するのに適した "
        f"主要 UI 要素（ボタン、ツールアイコン、トグル等）を最大 {max_elements} 個特定してください。\n\n"
        f"選定基準:\n"
        f"- 視覚的に明確で独自性のある要素（アイコン、ボタン）を優先\n"
        f"- テキストラベルだけの要素、メニュー項目（キーボードでアクセス可能）は除外\n"
        f"- 大きすぎる領域（パネル全体等）は除外\n"
        f"- よく使う操作に関連する要素を優先\n\n"
        f"JSON配列のみ返してください（マークダウン記法なし）:\n"
        f'[{{"name": "英語snake_case名", "description": "日本語の視覚的説明"}}, ...]\n\n'
        f"name は短く（例: pen_tool, save_button, layer_panel_toggle）\n"
        f"description はLLMが画面上で見つけられる視覚的な説明（例: 左ツールバーのペン型アイコン）"
        f"{knowledge_section}"
    )

    payload = {
        "model": model,
        "max_tokens": 1024,
        "system": system,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64_screenshot,
                    },
                },
                {"type": "text", "text": f"{app_name} のスクリーンショットです。主要 UI 要素を特定してください。"},
            ],
        }],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        import requests as _req
        resp = _req.post(
            "https://api.anthropic.com/v1/messages",
            json=payload, headers=headers, timeout=60,
        )
        if resp.status_code != 200:
            logger.warning(f"UI element identification API error: {resp.status_code} {resp.text[:200]}")
            return []

        text = resp.json()["content"][0]["text"].strip()

        import json as _json
        if "```" in text:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                text = text[start:end]
        elements = _json.loads(text)

        result = []
        for el in elements:
            if isinstance(el, dict) and "name" in el and "description" in el:
                result.append({
                    "name": str(el["name"]).strip(),
                    "description": str(el["description"]).strip(),
                })
        return result[:max_elements]

    except Exception as e:
        logger.warning(f"UI element identification failed: {e}")
        return []


def auto_learn_templates(
    screenshot: np.ndarray,
    app: str,
    api_key: str,
    matcher: "TemplateMatcher",
    app_knowledge: str = "",
    max_elements: int = 10,
    force: bool = False,
    grid_size: int = 5,
    max_steps: int = 4,
    target_px: int = 60,
    model: str = "claude-sonnet-4-20250514",
    monitor_phys_w: int = 3840,
    screenshot_w: int = 1920,
) -> dict:
    """スクリーンショットから主要 UI 要素を自動で一括学習する。"""
    t0 = time.time()

    b64 = _image_to_base64_jpeg(screenshot)

    elements = _identify_ui_elements(
        b64, app, api_key,
        app_knowledge=app_knowledge,
        max_elements=max_elements,
        model=model,
    )

    if not elements:
        return {
            "success": False, "app": app,
            "total_identified": 0, "skipped_existing": 0,
            "learned": [], "failed": [],
            "total_elapsed": time.time() - t0,
        }

    existing = set()
    all_templates = matcher.list_templates(app)
    for tmpl_list in all_templates.values():
        for tmpl in tmpl_list:
            existing.add(tmpl["name"])

    learned = []
    failed = []
    skipped = 0

    for el in elements:
        name = el["name"]
        desc = el["description"]

        if name in existing and not force:
            skipped += 1
            continue

        result = learn_template(
            screenshot=screenshot,
            target_description=desc,
            app=app,
            name=name,
            api_key=api_key,
            matcher=matcher,
            grid_size=grid_size,
            max_steps=max_steps,
            target_px=target_px,
            model=model,
            monitor_phys_w=monitor_phys_w,
            screenshot_w=screenshot_w,
        )

        if result["success"]:
            learned.append({
                "name": name, "description": desc,
                "path": result["path"],
                "steps": result["steps"],
                "elapsed": result["elapsed"],
            })
        else:
            failed.append({
                "name": name, "description": desc,
                "error": result["error"],
            })

    return {
        "success": len(learned) > 0,
        "app": app,
        "total_identified": len(elements),
        "skipped_existing": skipped,
        "learned": learned,
        "failed": failed,
        "total_elapsed": time.time() - t0,
    }
