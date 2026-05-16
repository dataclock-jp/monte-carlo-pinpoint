"""
llm_vision.py
LLM Vision APIを使って画面内容を解析し、次のアクションを提案するモジュール

Manus組み込みのForge API（OpenAI互換）を使用する。
環境変数 BUILT_IN_FORGE_API_URL / BUILT_IN_FORGE_API_KEY が設定されている場合は
それを優先し、なければ OPENAI_API_KEY / OPENAI_API_BASE を使用する。
"""
import os
import json
import base64
import requests
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ActionSuggestion:
    """LLMが提案する次のアクション。"""
    action_type: str          # "click" | "double_click" | "drag" | "type" | "scroll" | "keypress" | "wait" | "none"
    target: str               # 操作対象の説明（例: "送信ボタン"）
    value: str                # 入力値（typeの場合）またはキー名（keypressの場合）
    x: Optional[int]          # クリック座標X（click/scroll/dragの場合） — grid_cellから自動計算
    y: Optional[int]          # クリック座標Y（click/scroll/dragの場合） — grid_cellから自動計算
    confidence: float         # 確信度（0.0〜1.0）
    reasoning: str            # 提案の理由
    analysis: str             # 画面の状況説明
    grid_cell: str = ""       # グリッドセル名（例: "D3", "D3.7"）
    end_x: Optional[int] = None  # ドラッグ終了座標X — grid_cell_endから自動計算
    end_y: Optional[int] = None  # ドラッグ終了座標Y — grid_cell_endから自動計算
    grid_cell_end: str = ""      # ドラッグ終了グリッドセル名（例: "F4.3"）


@dataclass
class PlannedStep:
    """計画された将来のステップ。"""
    step: int                 # ステップ番号（1-indexed）
    action: str               # アクション種別
    target: str               # 操作対象の説明
    grid_cell: str = ""       # 対象グリッドセル（クリック系のみ）
    value: str = ""           # 入力値（type/keypressの場合）


@dataclass
class VisionAnalysis:
    """LLMによる画面解析結果。"""
    description: str          # 画面の状況説明
    changes: str              # 前フレームからの変化内容
    suggested_action: ActionSuggestion
    raw_response: str = ""
    planned_steps: list = field(default_factory=list)  # List[PlannedStep]


class LLMVision:
    """
    LLM Vision APIを使って画面内容を解析するクラス。
    
    画面スクリーンショット（Base64）を受け取り、
    - 画面の状況説明
    - 前フレームからの変化内容
    - 次に取るべきアクションの提案（座標付き）
    を返す。
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        language: str = "ja",
    ):
        """
        Parameters
        ----------
        model : str
            使用するLLMモデル名
        api_url : str, optional
            API URL（Noneの場合は環境変数から取得）
        api_key : str, optional
            APIキー（Noneの場合は環境変数から取得）
        language : str
            応答言語（"ja" or "en"）
        """
        self.model = model
        self.language = language
        self._is_claude = model.startswith("claude")

        if self._is_claude:
            self.api_url = "https://api.anthropic.com"
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        else:
            self.api_url = api_url or os.environ.get("BUILT_IN_FORGE_API_URL") or os.environ.get("OPENAI_API_BASE", "https://api.openai.com")
            self.api_key = api_key or os.environ.get("BUILT_IN_FORGE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

        if not self.api_key:
            raise ValueError("API key not found.")

    def _call_api(self, system_prompt: str, user_prompt: str, b64_data: str) -> str:
        """LLM APIを呼び出してレスポンステキストを返す。Claude/OpenAI互換を自動切替。"""
        if self._is_claude:
            return self._call_anthropic(system_prompt, user_prompt, b64_data)
        else:
            return self._call_openai_compat(system_prompt, user_prompt, b64_data)

    def _call_anthropic(self, system_prompt: str, user_prompt: str, b64_data: str) -> str:
        """Anthropic Messages APIを呼び出す。"""
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64_data,
                            },
                        },
                        {"type": "text", "text": user_prompt + "\n\nJSON形式のみで回答してください。他のテキストは含めないでください。"},
                    ],
                },
            ],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        resp = requests.post(
            f"{self.api_url}/v1/messages",
            json=payload,
            headers=headers,
            timeout=60,
        )
        if resp.status_code != 200:
            detail = resp.text[:500]
            raise Exception(f"{resp.status_code} Anthropic API error: {detail}")
        result = resp.json()
        return result["content"][0]["text"]

    def _call_openai_compat(self, system_prompt: str, user_prompt: str, b64_data: str) -> str:
        """OpenAI互換APIを呼び出す。"""
        image_url = f"data:image/jpeg;base64,{b64_data}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url, "detail": "high"},
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "screen_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "changes": {"type": "string"},
                            "action_type": {"type": "string"},
                            "target": {"type": "string"},
                            "value": {"type": "string"},
                            "grid_cell": {"type": "string"},
                            "grid_cell_end": {"type": "string"},
                            "confidence": {"type": "number"},
                            "reasoning": {"type": "string"},
                            "planned_steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "step": {"type": "integer"},
                                        "action": {"type": "string"},
                                        "target": {"type": "string"},
                                        "grid_cell": {"type": "string"},
                                        "value": {"type": "string"},
                                    },
                                    "required": ["step", "action", "target", "grid_cell", "value"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["description", "changes", "action_type", "target",
                                     "value", "grid_cell", "grid_cell_end", "confidence",
                                     "reasoning", "planned_steps"],
                        "additionalProperties": False,
                    },
                },
            },
            "max_tokens": 1024,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            f"{self.api_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=30,
        )
        if resp.status_code != 200:
            detail = resp.text[:500]
            raise Exception(f"{resp.status_code} API error: {detail}")
        result = resp.json()
        return result["choices"][0]["message"]["content"]

    def analyze(
        self,
        current_frame_b64: str,
        goal: str = "",
        prev_analysis: Optional[str] = None,
        diff_score: float = 0.0,
        screen_size: tuple = (1920, 1080),
        working_memory_text: str = "",
        long_term_memory_text: str = "",
    ) -> VisionAnalysis:
        """
        スクリーンショットを解析して次のアクションを提案する。

        Parameters
        ----------
        current_frame_b64 : str
            現在のスクリーンショット（data:image/jpeg;base64,...形式）
        goal : str
            エージェントの目標（例: "ファイルを保存する"）
        prev_analysis : str, optional
            前回の解析結果（文脈として渡す）
        diff_score : float
            変化スコア（0.0〜1.0）
        screen_size : tuple
            スクリーンサイズ (width, height)
        working_memory_text : str
            作業記憶のプロンプト用テキスト（空文字列なら省略）
        long_term_memory_text : str
            長期記憶のプロンプト用テキスト（空文字列なら省略）

        Returns
        -------
        VisionAnalysis
            解析結果と次のアクション提案
        """
        lang_instruction = "日本語で回答してください。" if self.language == "ja" else "Please respond in English."
        goal_text = f"\n目標: {goal}" if goal else ""
        prev_text = f"\n前回の状況: {prev_analysis}" if prev_analysis else ""
        screen_w, screen_h = screen_size

        system_prompt = f"""あなたはデスクトップ操作を支援するAIエージェントです。
スクリーンショットを分析し、画面の状況と「目標達成のために今必要な1つの操作」を提案してください。
{lang_instruction}

■ グリッドによる位置指定
画像には緑色のグリッド（8列A-H × 5行1-5）がオーバーレイされています。
各セルの左上にラベル（例: A1, D3, H5）が表示されています。
クリックやスクロールの対象位置は、ピクセル座標ではなく **グリッドセル名**で指定してください。

■ サブグリッド（精密な位置指定）
セル中央以外を狙いたい場合、テンキー配列でセル内の位置を指定できます:
  7(左上)  8(上)   9(右上)
  4(左)    5(中央)  6(右)
  1(左下)  2(下)   3(右下)
書式: "D3.7" = D3セルの左上、"D3.5" = D3セルの中央（"D3" と同じ）
画像上の緑色ドットがサブグリッド位置を示しています。
小さなボタンやリンクを正確にクリックしたい場合に使ってください。

画像上の赤い十字マーカー（⊕）は現在のマウスカーソル位置を示しています。
変化スコア: {diff_score:.3f}（0=変化なし, 1=大きな変化）{goal_text}{prev_text}{working_memory_text}{long_term_memory_text}

重要なルール:
- 目標を段階的に進めてください。同じ操作を2回以上繰り返さないでください。
- アプリを起動した後は、起動完了を待ってから次の操作に進んでください（waitアクション）。
- 目標が全て完了したら action_type を "none" にしてください。
- "none" を返す前に、目標の全ステップが実際に完了しているか画面で確認してください。特にテキスト入力が目標に含まれる場合、入力したテキストが画面上に表示されていなければまだ完了していません。

操作の優先順位（キーボード操作を優先すること）:
- 特定のウィンドウにフォーカスしたい場合は focus_window を使ってください（valueにウィンドウタイトルのキーワードを指定。例: "メモ帳", "Notepad", "Chrome"）。Alt+Tabより確実です。
- Alt+Tabは直前のウィンドウに戻る場合にのみ使ってください。特定のウィンドウを探すためにAlt+Tabを繰り返さないでください。
- アプリ起動後、そのアプリが前面にあれば直接typeで入力できます。クリックは不要です。
- テキスト入力欄へのフォーカスにはTabキーを使ってください。
- clickは、キーボード操作では到達できない場合の最終手段です。
- clickした後に変化スコアが低い（≈0）なら、クリックは成功しています。同じclickを繰り返さず、次のアクション（keypress/type）に進んでください。
- テキスト入力後: 入力したテキストが画面上に正しく表示されたか確認してください。表示されていなければ、入力先が間違っています。Alt+Tabで正しいウィンドウに切り替えてから再入力してください。
- 画面上のテキストが読みにくい場合は、コンテキストに含まれるOCR結果を参照してください。OCR結果はスクリーンショットよりも正確にテキストを認識しています。

禁止操作（絶対に使わないこと）:
- win+d（デスクトップ表示）、win+m（全ウィンドウ最小化）は禁止。ユーザーの作業が破壊されます。
- alt+F4（ウィンドウを閉じる）は禁止。目的のアプリを閉じてしまいます。

Windows 11の注意点:
- メモ帳（Notepad）はダークテーマに対応しており、黒い背景のテキストエディタとして表示されます。VSCodeやSublime Textと間違えないでください。タイトルバーに「タイトルなし」「無題」「Untitled」等と表示されていればメモ帳です。
- 検索（Win+S）の結果からEnterキーでアプリを起動した後、自動でAlt+Tabが実行されフォーカスが移動します。アプリが前面に表示されていれば、クリックせずにそのままtypeで入力してください。

■ 操作計画
目標達成に向けた今後2〜3ステップの計画を planned_steps に記述してください。
各ステップは、action（操作種別）、target（対象）、grid_cell（クリック系のみ）、value（入力値）を含みます。
計画を立てることで、見通しの良い操作が可能になります。"""

        user_prompt = """このスクリーンショットを分析して、以下のJSON形式で回答してください:

{
  "description": "現在の画面状況の説明（何が表示されているか）",
  "changes": "前フレームからの変化内容（初回は「初期状態」）",
  "action_type": "次に取るべき操作（click/double_click/drag/type/scroll/keypress/focus_window/wait/none のいずれか）",
  "target": "操作対象の説明（例: 「送信ボタン」「検索フィールド」）",
  "value": "入力テキストまたはキー名（typeの場合は入力する文字列全体、keypressでは win+s, ctrl+s, Return, Escape 等のpyautogui形式で指定）",
  "grid_cell": "クリック/スクロール/ドラッグ開始位置のグリッドセル名（例: 'D3', 'D3.7'）。click/double_click/scroll/dragの場合は必須。それ以外は空文字。",
  "grid_cell_end": "ドラッグ終了位置のグリッドセル名。dragの場合のみ必須（例: 'F4.3'）。それ以外は空文字。",
  "confidence": 確信度（0.0〜1.0の小数）,
  "reasoning": "この操作を提案する理由（目標のどの段階にいるかを含めて）",
  "planned_steps": [
    {"step": 1, "action": "操作種別", "target": "対象", "grid_cell": "セル名(クリック系のみ)", "value": "入力値(type/keypressのみ)"},
    {"step": 2, "action": "...", "target": "...", "grid_cell": "", "value": ""}
  ]
}

重要: クリック/スクロール/ドラッグ位置はグリッドセル名（A1〜H5、サブグリッド付きはA1.7〜H5.3）で指定してください。ピクセル座標は不要です。
planned_steps には、現在のアクションの後に続く2〜3ステップの計画を記述してください。"""

        # Base64生データを取得
        if current_frame_b64.startswith("data:"):
            b64_data = current_frame_b64.split(",", 1)[1]
        else:
            b64_data = current_frame_b64

        raw_content = self._call_api(system_prompt, user_prompt, b64_data)

        # markdownコードブロックを除去（```json ... ``` 対策）
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            # 最初の行（```json等）を除去
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1:]
            # 末尾の ``` を除去
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # JSON解析失敗時のフォールバック
            parsed = {
                "description": raw_content[:200],
                "changes": "解析失敗",
                "action_type": "none",
                "target": "",
                "value": "",
                "x": 0,
                "y": 0,
                "confidence": 0.0,
                "reasoning": "JSON解析に失敗しました",
            }

        grid_cell = str(parsed.get("grid_cell", "")).strip().upper()
        grid_cell_end = str(parsed.get("grid_cell_end", "")).strip().upper()

        # grid_cell → 画像座標に変換
        img_x: Optional[int] = None
        img_y: Optional[int] = None
        if grid_cell and len(grid_cell) >= 2 and grid_cell[0].isalpha():
            from agent.screen_capture import grid_cell_to_image_coords
            img_x, img_y = grid_cell_to_image_coords(grid_cell, screen_w, screen_h)

        # grid_cell_end → ドラッグ終了座標に変換
        end_x: Optional[int] = None
        end_y: Optional[int] = None
        if grid_cell_end and len(grid_cell_end) >= 2 and grid_cell_end[0].isalpha():
            from agent.screen_capture import grid_cell_to_image_coords
            end_x, end_y = grid_cell_to_image_coords(grid_cell_end, screen_w, screen_h)

        # フォールバック: LLMが従来のx/yを返した場合も受け付ける
        if img_x is None and int(parsed.get("x", 0)):
            img_x = int(parsed["x"])
            img_y = int(parsed.get("y", 0))

        action = ActionSuggestion(
            action_type=str(parsed.get("action_type", "none")),
            target=str(parsed.get("target", "")),
            value=str(parsed.get("value", "")),
            x=img_x,
            y=img_y,
            confidence=float(parsed.get("confidence", 0.0)),
            reasoning=str(parsed.get("reasoning", "")),
            analysis=str(parsed.get("description", "")),
            grid_cell=grid_cell,
            end_x=end_x,
            end_y=end_y,
            grid_cell_end=grid_cell_end,
        )

        # planned_steps をパース
        planned_steps = []
        for ps in parsed.get("planned_steps", []):
            if isinstance(ps, dict):
                planned_steps.append(PlannedStep(
                    step=int(ps.get("step", 0)),
                    action=str(ps.get("action", "")),
                    target=str(ps.get("target", "")),
                    grid_cell=str(ps.get("grid_cell", "")),
                    value=str(ps.get("value", "")),
                ))

        return VisionAnalysis(
            description=str(parsed.get("description", "")),
            changes=str(parsed.get("changes", "")),
            suggested_action=action,
            raw_response=raw_content,
            planned_steps=planned_steps,
        )

    # ------------------------------------------------------------------
    # 並列ワーカー用メソッド
    # ------------------------------------------------------------------

    def analyze_observation_only(
        self,
        current_frame_b64: str,
        diff_score: float = 0.0,
        screen_size: tuple = (1920, 1080),
    ) -> dict:
        """
        画面の状況を記述するのみ。アクション提案は行わない（ワーカー用）。

        Returns
        -------
        dict
            {"description": str, "changes": str}
        """
        lang = "日本語で回答してください。" if self.language == "ja" else "Respond in English."
        screen_w, screen_h = screen_size

        system_prompt = f"""あなたは画面観察専門のAIです。スクリーンショットを見て、画面の状況と変化を簡潔に記述してください。
アクションの提案は絶対にしないでください。観察結果のみを返してください。
{lang}
スクリーン解像度: {screen_w}x{screen_h}
変化スコア: {diff_score:.3f}"""

        user_prompt = "このスクリーンショットの状況と、前回からの変化を記述してください。"

        if current_frame_b64.startswith("data:"):
            image_url = current_frame_b64
        else:
            image_url = f"data:image/jpeg;base64,{current_frame_b64}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "screen_observation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "changes": {"type": "string"},
                        },
                        "required": ["description", "changes"],
                        "additionalProperties": False,
                    },
                },
            },
            "max_tokens": 512,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(
            f"{self.api_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()

        raw = resp.json()["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"description": raw[:200], "changes": "解析失敗"}

        return {
            "description": str(parsed.get("description", "")),
            "changes": str(parsed.get("changes", "")),
        }

    def decide_action(
        self,
        reports: list[dict],
        goal: str = "",
        current_frame_b64: Optional[str] = None,
        screen_size: tuple = (1920, 1080),
    ) -> ActionSuggestion:
        """
        ワーカーからの観察報告を集約し、1つのアクションを決定する（コーディネーター用）。

        Parameters
        ----------
        reports : list[dict]
            ワーカーからの観察報告リスト。各要素に description, changes キー。
        goal : str
            エージェントの目標。
        current_frame_b64 : str, optional
            最新のスクリーンショット（コーディネーターの視覚的根拠用）。
        screen_size : tuple
            スクリーンサイズ。

        Returns
        -------
        ActionSuggestion
        """
        lang = "日本語で回答してください。" if self.language == "ja" else "Respond in English."
        screen_w, screen_h = screen_size
        goal_text = f"\n目標: {goal}" if goal else ""

        # 観察報告をテキストに整形
        report_lines = []
        for i, r in enumerate(reports):
            report_lines.append(f"[報告{i+1}] {r.get('description', '')} | 変化: {r.get('changes', '')}")
        reports_text = "\n".join(report_lines)

        system_prompt = f"""あなたはデスクトップ操作の統括AIです。
画面観察ワーカーからの報告を受け取り、目標に照らして最も適切な「1つのアクション」を決定してください。
複数のアクションが必要な場合でも、今この瞬間に実行すべき1つだけを選んでください。
{lang}
スクリーン解像度: {screen_w}x{screen_h}{goal_text}"""

        user_prompt = f"""以下はワーカーからの画面観察報告です:

{reports_text}

この報告に基づき、次に取るべき1つのアクションをJSON形式で返してください。"""

        # メッセージ構築
        content: list[dict] = []
        if current_frame_b64:
            image_url = current_frame_b64 if current_frame_b64.startswith("data:") else f"data:image/jpeg;base64,{current_frame_b64}"
            content.append({"type": "image_url", "image_url": {"url": image_url, "detail": "high"}})
        content.append({"type": "text", "text": user_prompt})

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "action_decision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "action_type": {"type": "string"},
                            "target": {"type": "string"},
                            "value": {"type": "string"},
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                            "confidence": {"type": "number"},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["action_type", "target", "value", "x", "y", "confidence", "reasoning"],
                        "additionalProperties": False,
                    },
                },
            },
            "max_tokens": 512,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(
            f"{self.api_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()

        raw = resp.json()["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"action_type": "none", "target": "", "value": "",
                      "x": 0, "y": 0, "confidence": 0.0, "reasoning": "JSON解析失敗"}

        return ActionSuggestion(
            action_type=str(parsed.get("action_type", "none")),
            target=str(parsed.get("target", "")),
            value=str(parsed.get("value", "")),
            x=int(parsed.get("x", 0)) or None,
            y=int(parsed.get("y", 0)) or None,
            confidence=float(parsed.get("confidence", 0.0)),
            reasoning=str(parsed.get("reasoning", "")),
            analysis="",
        )
