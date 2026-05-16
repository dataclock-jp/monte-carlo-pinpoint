"""
system2.py
System 2 思考ループ（熟慮的推論パイプライン）

Kahneman の二重過程理論に基づく、遅い思考（deliberate reasoning）の実装。
知覚 → 作業記憶更新 → 脳内シミュレーション → 弁証法チェック → 行動決定
の完全パイプラインを提供する。

通常のエージェントループ（System 1 的な高速判断）で対処できない場面で、
ゲートキーパーが System 2 への切り替えを判定し、このパイプラインが起動する。

トリガー条件:
  - 低確信度（< 0.7）
  - 連続失敗（2回以上）
  - 予測誤差（画面変化が想定外）
  - 高複雑度タスク（5ステップ以上の計画）
  - ユーザー明示的要求

停止基準（overthinking 防止）:
  - 最大反復回数（5回）
  - タイムアウト（30秒）
  - 確信度閾値到達（0.85）
  - 結論安定（2回連続同一結論）

参考文献:
  - Kahneman, "Thinking, Fast and Slow", 2011
  - Google DeepMind, "Talker-Reasoner Architecture", arXiv:2410.08328
  - CogniGUI, arXiv:2506.17913
  - DPT-Agent, arXiv:2502.11882 (ACL 2025)
  - D-Mem, arXiv:2603.18631
"""
import time
import json
import hashlib
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum
from pathlib import Path


class ThinkingMode(Enum):
    """思考モード。"""
    SYSTEM1 = "system1"  # 高速・直感的
    SYSTEM2 = "system2"  # 低速・熟慮的


@dataclass
class System2Result:
    """System 2 思考ループの結果。"""
    action_type: str           # 推奨アクション種別
    target: str                # 操作対象
    value: str                 # 入力値
    grid_cell: str             # グリッドセル
    confidence: float          # 最終確信度
    reasoning: str             # 推論の根拠
    iterations: int            # 思考ループの反復回数
    duration_seconds: float    # 思考にかかった時間
    exit_reason: str           # 終了理由（"confidence", "stable", "timeout", "max_iter"）
    dialectic_summary: str     # 弁証法的検証の要約
    simulation_result: str     # キャンバスシミュレーションの要約


@dataclass
class GatekeeperContext:
    """ゲートキーパーの判定に使うコンテキスト。"""
    confidence: float = 1.0
    last_action_failed: bool = False
    consecutive_failures: int = 0
    screen_diff_unexpected: bool = False
    estimated_steps: int = 1
    working_memory_full: bool = False
    is_first_action: bool = False
    is_risky_action: bool = False


class Gatekeeper:
    """
    System 1/2 の切り替えを判定するルーター。

    多次元ゲーティングで、確信度・新規性・失敗履歴・複雑性を
    総合的に評価し、System 2 の起動が必要かどうかを判定する。
    """

    # 閾値設定
    CONFIDENCE_THRESHOLD = 0.7
    FAILURE_THRESHOLD = 2

    def judge(self, ctx: GatekeeperContext) -> ThinkingMode:
        """コンテキストから思考モードを判定する。"""
        # 低確信度
        if ctx.confidence < self.CONFIDENCE_THRESHOLD:
            return ThinkingMode.SYSTEM2

        # 連続失敗
        if ctx.consecutive_failures >= self.FAILURE_THRESHOLD:
            return ThinkingMode.SYSTEM2

        # 予測誤差（画面変化が想定外）
        if ctx.screen_diff_unexpected:
            return ThinkingMode.SYSTEM2

        # 高複雑度
        if ctx.estimated_steps >= 5:
            return ThinkingMode.SYSTEM2

        # 初回アクション（状況把握が必要）
        if ctx.is_first_action:
            return ThinkingMode.SYSTEM2

        # リスクの高い操作
        if ctx.is_risky_action:
            return ThinkingMode.SYSTEM2

        # WM が満杯（認知負荷が高い）
        if ctx.working_memory_full:
            return ThinkingMode.SYSTEM2

        return ThinkingMode.SYSTEM1

    def build_context(self, suggestion, action_log: list,
                      working_memory, step: int,
                      diff_score: float = 0.0,
                      expected_diff: float = 0.0) -> GatekeeperContext:
        """エージェントの状態から GatekeeperContext を構築する。"""
        # 連続失敗数をカウント
        consecutive_failures = 0
        for a in reversed(action_log):
            if not a.get("success", True):
                consecutive_failures += 1
            else:
                break

        # リスクの高い操作の判定
        risky_keys = {"ctrl+s", "ctrl+shift+s", "delete", "ctrl+x",
                      "ctrl+z", "ctrl+n", "ctrl+shift+n"}
        is_risky = False
        if suggestion.action_type == "keypress":
            key = suggestion.value.lower().replace(" ", "")
            is_risky = key in risky_keys

        # 計画ステップ数の推定（planned_steps があれば）
        estimated_steps = 1
        if hasattr(suggestion, '_planned_steps_count'):
            estimated_steps = suggestion._planned_steps_count

        # 予測誤差
        screen_diff_unexpected = False
        if expected_diff > 0 and diff_score > 0:
            prediction_error = abs(diff_score - expected_diff) / max(expected_diff, 0.01)
            screen_diff_unexpected = prediction_error > 2.0  # 2倍以上のずれ

        return GatekeeperContext(
            confidence=suggestion.confidence,
            last_action_failed=consecutive_failures > 0,
            consecutive_failures=consecutive_failures,
            screen_diff_unexpected=screen_diff_unexpected,
            estimated_steps=estimated_steps,
            working_memory_full=len(working_memory) >= working_memory.capacity,
            is_first_action=(step == 1),
            is_risky_action=is_risky,
        )


class System2Loop:
    """
    System 2 思考ループの実行エンジン。

    知覚 → WM更新 → 脳内シミュレーション → 弁証法チェック → 行動決定
    のパイプラインを反復し、確信度が十分高くなるか停止基準に達するまで
    思考を深める。
    """

    MAX_ITERATIONS = 5
    TIMEOUT_SECONDS = 30
    CONFIDENCE_EXIT = 0.85

    def __init__(self, working_memory, canvas_mgr, dialectic):
        self.working_memory = working_memory
        self.canvas_mgr = canvas_mgr
        self.dialectic = dialectic
        self._iteration_history: List[dict] = []
        self._depth_controller = ThinkingDepthController()

    def run(self, suggestion, analysis_desc: str,
            screen_capture_fn=None, monitor_index: int = 1,
            gatekeeper_ctx: GatekeeperContext = None) -> System2Result:
        """
        System 2 思考ループを実行する。

        Parameters
        ----------
        suggestion : ActionSuggestion
            System 1 が提案したアクション
        analysis_desc : str
            画面の状況説明
        screen_capture_fn : callable, optional
            スクリーンショット取得関数（Canvas シミュレーション用）
        monitor_index : int
            キャプチャ対象モニター
        gatekeeper_ctx : GatekeeperContext, optional
            ゲートキーパーのコンテキスト（思考深度の動的調整に使用）

        Returns
        -------
        System2Result
            熟慮の結果
        """
        start = time.time()
        self._iteration_history.clear()
        prev_conclusion = None
        best_result = None
        session_name = f"s2_{int(start)}"

        # 思考深度の動的調整
        if gatekeeper_ctx:
            confidence_history = [h.get("confidence", 0.5)
                                  for h in self._iteration_history]
            max_iter = self._depth_controller.calculate_depth(
                gatekeeper_ctx, confidence_history)
        else:
            max_iter = self.MAX_ITERATIONS

        # テーゼ: System 1 の提案
        thesis = (f"{suggestion.action_type}({suggestion.target}) "
                  f"[確信度={suggestion.confidence:.0%}]: {suggestion.reasoning}")

        for iteration in range(max_iter):
            elapsed = time.time() - start

            # タイムアウトチェック
            if elapsed > self.TIMEOUT_SECONDS:
                return self._build_result(
                    suggestion, iteration, elapsed, "timeout",
                    best_result or {})

            # Phase 1: 作業記憶の確認
            wm_context = self.working_memory.recall()

            # Phase 2: 脳内シミュレーション（Canvas）
            simulation_result = ""
            if screen_capture_fn and suggestion.grid_cell:
                simulation_result = self._simulate_on_canvas(
                    suggestion, monitor_index)

            # Phase 3: 弁証法チェック
            if iteration == 0:
                # 初回: テーゼとアンチテーゼを設定
                self.dialectic.start(session_name,
                                     goal="最適な操作を判断する",
                                     thesis=thesis)
                antithesis = self._generate_antithesis(
                    suggestion, wm_context, simulation_result)
                self.dialectic.challenge(session_name, antithesis)
                dialectic_summary = f"テーゼ: {thesis}\nアンチテーゼ: {antithesis}"
            else:
                # 2回目以降: 前回のジンテーゼを新テーゼとして反復
                if prev_conclusion:
                    triad = self.dialectic.iterate(session_name)
                    if triad:
                        new_anti = self._refine_antithesis(
                            prev_conclusion, wm_context, iteration)
                        self.dialectic.challenge(session_name, new_anti)
                        dialectic_summary = f"新テーゼ: {prev_conclusion}\nアンチテーゼ: {new_anti}"
                    else:
                        dialectic_summary = "反復不可（前ラウンド未完了）"
                else:
                    dialectic_summary = ""

            # Phase 4: ジンテーゼ（統合判断）
            synthesis = self._synthesize(
                suggestion, wm_context, simulation_result, dialectic_summary)
            self.dialectic.synthesize(session_name, synthesis["conclusion"])

            # 反復記録
            iter_record = {
                "iteration": iteration + 1,
                "confidence": synthesis["confidence"],
                "conclusion": synthesis["conclusion"],
                "elapsed": time.time() - start,
            }
            self._iteration_history.append(iter_record)
            best_result = synthesis

            # 停止判定: 確信度
            if synthesis["confidence"] >= self.CONFIDENCE_EXIT:
                self.dialectic.conclude(session_name, synthesis["conclusion"])
                return self._build_result(
                    suggestion, iteration + 1, time.time() - start,
                    "confidence", synthesis,
                    dialectic_summary, simulation_result)

            # 停止判定: 結論安定（2回連続同一）
            if prev_conclusion and prev_conclusion == synthesis["conclusion"]:
                self.dialectic.conclude(session_name, synthesis["conclusion"])
                return self._build_result(
                    suggestion, iteration + 1, time.time() - start,
                    "stable", synthesis,
                    dialectic_summary, simulation_result)

            prev_conclusion = synthesis["conclusion"]

        # 最大反復到達
        self.dialectic.conclude(session_name)
        return self._build_result(
            suggestion, max_iter, time.time() - start,
            "max_iter", best_result or {},
            dialectic_summary, simulation_result)

    def _simulate_on_canvas(self, suggestion, monitor_index: int) -> str:
        """キャンバス上で操作結果をシミュレートする。"""
        try:
            from agent.screen_capture import grid_cell_to_image_coords
            canvas = self.canvas_mgr.from_screenshot(
                "_s2_sim", monitor=monitor_index)

            gc = suggestion.grid_cell.strip().upper()
            if len(gc) >= 2 and gc[0].isalpha():
                # 仮のスクリーンサイズ（キャンバスのサイズを使用）
                img = canvas.get_image()
                x, y = grid_cell_to_image_coords(gc, img.width, img.height)
                if x is not None:
                    canvas.draw_marker(x, y, label=f"S2: {suggestion.action_type}",
                                       color="red", size=25)
                    canvas.snapshot("s2_planned")
                    result = f"シミュレーション: {suggestion.action_type} at {gc} ({x},{y}) をマーク済み"
                else:
                    result = f"シミュレーション: グリッドセル {gc} の座標変換失敗"
            else:
                result = f"シミュレーション: グリッドセルなし（{suggestion.action_type}）"

            self.canvas_mgr.delete("_s2_sim")
            return result
        except Exception as e:
            return f"シミュレーション失敗: {e}"

    def _generate_antithesis(self, suggestion, wm_context: str,
                             simulation_result: str) -> str:
        """初回のアンチテーゼを生成する。"""
        challenges = []

        # 確信度への疑問
        if suggestion.confidence < 0.7:
            challenges.append(
                f"確信度が{suggestion.confidence:.0%}と低い。"
                f"対象「{suggestion.target}」の識別は正確か？")

        # 作業記憶からの矛盾
        if wm_context:
            lines = wm_context.split("\n")
            same_action = [l for l in lines if suggestion.action_type in l]
            if len(same_action) >= 2:
                challenges.append(
                    f"{suggestion.action_type}が直近で{len(same_action)}回実行済み。"
                    f"繰り返しに効果はあるか？")
            if any("失敗" in l for l in lines[-2:]):
                challenges.append(
                    "直前に失敗が記録されている。同じアプローチは妥当か？")

        # クリック系への代替案提案
        if suggestion.action_type in ("click", "double_click"):
            challenges.append(
                f"クリック({suggestion.grid_cell})の代わりに"
                f"キーボード操作（Tab, ショートカット）で代替できないか？")

        # シミュレーション結果
        if "失敗" in simulation_result:
            challenges.append(f"脳内シミュレーション結果: {simulation_result}")

        if not challenges:
            challenges.append("この操作が最短経路か？別の効率的な方法はないか？")

        return " ".join(challenges)

    def _refine_antithesis(self, prev_conclusion: str, wm_context: str,
                           iteration: int) -> str:
        """反復ごとに洗練されたアンチテーゼを生成する。"""
        return (f"前回の結論「{prev_conclusion[:60]}」は十分に検証されたか？"
                f"見落としている前提条件やリスクはないか？"
                f"（反復{iteration + 1}回目）")

    def _synthesize(self, suggestion, wm_context: str,
                    simulation_result: str, dialectic_summary: str) -> dict:
        """テーゼ・アンチテーゼ・シミュレーション結果を統合してジンテーゼを生成する。"""
        # 確信度の再計算
        confidence = suggestion.confidence

        # 作業記憶に矛盾がなければ +0.1
        if wm_context and "失敗" not in wm_context:
            confidence = min(1.0, confidence + 0.1)

        # シミュレーション成功で +0.05
        if simulation_result and "失敗" not in simulation_result:
            confidence = min(1.0, confidence + 0.05)

        # 弁証法で重大な反論がなければ +0.05
        if dialectic_summary and "繰り返し" not in dialectic_summary:
            confidence = min(1.0, confidence + 0.05)

        conclusion = (f"{suggestion.action_type}({suggestion.target}) を実行。"
                      f"確信度={confidence:.0%}。"
                      f"{suggestion.reasoning}")

        return {
            "confidence": confidence,
            "conclusion": conclusion,
        }

    def _build_result(self, suggestion, iterations: int,
                      duration: float, exit_reason: str,
                      synthesis: dict,
                      dialectic_summary: str = "",
                      simulation_result: str = "") -> System2Result:
        """System2Result を構築する。"""
        return System2Result(
            action_type=suggestion.action_type,
            target=suggestion.target,
            value=suggestion.value,
            grid_cell=getattr(suggestion, 'grid_cell', ''),
            confidence=synthesis.get("confidence", suggestion.confidence),
            reasoning=synthesis.get("conclusion", suggestion.reasoning),
            iterations=iterations,
            duration_seconds=duration,
            exit_reason=exit_reason,
            dialectic_summary=dialectic_summary,
            simulation_result=simulation_result,
        )


# ===========================================================================
# Phase 2: 経験キャッシュ（S2→S1 降格）
# ===========================================================================

@dataclass
class CachedPattern:
    """キャッシュされた操作パターン。"""
    task_signature: str           # タスクの署名（ゴール+コンテキストのハッシュ）
    action_sequence: List[dict]   # 成功した操作シーケンス
    success_count: int = 0        # 成功回数
    last_used: float = 0.0        # 最終使用タイムスタンプ
    created_at: float = 0.0       # 作成タイムスタンプ


class ExperienceCache:
    """
    System 2 の成功パターンを System 1 に降格するキャッシュ。

    Soar アーキテクチャの「チャンキング」、SOFAI の「経験ベース O(1) ソルバー」に
    相当する。繰り返し成功したパターンをキャッシュし、同一タスク再発時に
    System 2 をスキップして即座に実行する。

    昇格条件: 同一署名で PROMOTION_THRESHOLD 回以上成功
    永続化: JSON ファイルに保存（セッション跨ぎで再利用）
    容量制限: MAX_PATTERNS を超えると最も古い未使用パターンを削除
    """

    PROMOTION_THRESHOLD = 2   # この回数成功したらキャッシュに昇格
    MAX_PATTERNS = 100        # 最大パターン数

    def __init__(self, cache_dir: str = ""):
        self._patterns: Dict[str, CachedPattern] = {}
        self._pending: Dict[str, list] = {}  # 署名 → 進行中のアクションシーケンス
        if cache_dir:
            self._cache_path = Path(cache_dir) / "experience_cache.json"
        else:
            self._cache_path = (Path.home() / "video2ai_agent_sessions"
                                / "experience_cache.json")
        self._load()

    @staticmethod
    def make_signature(goal: str, app_context: str = "") -> str:
        """タスクの署名を生成する。"""
        raw = f"{goal.strip().lower()}|{app_context.strip().lower()}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def lookup(self, signature: str) -> Optional[CachedPattern]:
        """キャッシュされたパターンを検索する。昇格済みのもののみ返す。"""
        pattern = self._patterns.get(signature)
        if pattern and pattern.success_count >= self.PROMOTION_THRESHOLD:
            pattern.last_used = time.time()
            return pattern
        return None

    def begin_recording(self, signature: str) -> None:
        """アクションシーケンスの記録を開始する。"""
        self._pending[signature] = []

    def record_action(self, signature: str, action: dict) -> None:
        """進行中のシーケンスにアクションを追加する。"""
        if signature in self._pending:
            self._pending[signature].append({
                "action_type": action.get("action_type", ""),
                "target": action.get("target", ""),
                "value": action.get("value", ""),
                "grid_cell": action.get("grid_cell", ""),
            })

    def complete_success(self, signature: str) -> bool:
        """タスクの成功を記録し、キャッシュを更新する。昇格したら True を返す。"""
        sequence = self._pending.pop(signature, None)
        if not sequence:
            return False

        if signature in self._patterns:
            pattern = self._patterns[signature]
            pattern.success_count += 1
            pattern.last_used = time.time()
            # より短いシーケンスがあれば更新（効率化）
            if len(sequence) < len(pattern.action_sequence):
                pattern.action_sequence = sequence
        else:
            pattern = CachedPattern(
                task_signature=signature,
                action_sequence=sequence,
                success_count=1,
                last_used=time.time(),
                created_at=time.time(),
            )
            self._patterns[signature] = pattern

        promoted = pattern.success_count >= self.PROMOTION_THRESHOLD
        self._evict_if_needed()
        self._save()
        return promoted

    def complete_failure(self, signature: str) -> None:
        """タスクの失敗を記録し、記録中のシーケンスを破棄する。"""
        self._pending.pop(signature, None)

    def stats(self) -> dict:
        """キャッシュの統計情報を返す。"""
        promoted = sum(1 for p in self._patterns.values()
                       if p.success_count >= self.PROMOTION_THRESHOLD)
        return {
            "total_patterns": len(self._patterns),
            "promoted_patterns": promoted,
            "pending_recordings": len(self._pending),
        }

    def _evict_if_needed(self) -> None:
        """容量超過時に最も古い未使用パターンを削除する。"""
        while len(self._patterns) > self.MAX_PATTERNS:
            oldest_key = min(self._patterns,
                             key=lambda k: self._patterns[k].last_used)
            del self._patterns[oldest_key]

    def _save(self) -> None:
        """キャッシュを JSON ファイルに永続化する。"""
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for sig, p in self._patterns.items():
                data[sig] = {
                    "task_signature": p.task_signature,
                    "action_sequence": p.action_sequence,
                    "success_count": p.success_count,
                    "last_used": p.last_used,
                    "created_at": p.created_at,
                }
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load(self) -> None:
        """JSON ファイルからキャッシュを読み込む。"""
        try:
            if self._cache_path.exists():
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for sig, d in data.items():
                    self._patterns[sig] = CachedPattern(
                        task_signature=d["task_signature"],
                        action_sequence=d["action_sequence"],
                        success_count=d.get("success_count", 0),
                        last_used=d.get("last_used", 0),
                        created_at=d.get("created_at", 0),
                    )
        except Exception:
            pass


# ===========================================================================
# Phase 2: 予測誤差モニター
# ===========================================================================

class PredictionErrorMonitor:
    """
    行動の結果予測と実際の結果を比較し、「驚き」の度合いを定量化する。
    Kahneman の「世界モデル違反」検出に直接対応。

    高い予測誤差 → System 2 トリガー
    低い予測誤差 → System 1 で続行
    """

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        self._expected_diff: float = 0.0
        self._history: List[dict] = []

    def predict(self, action_type: str) -> float:
        """アクション種別から期待される画面変化量を予測する。"""
        # アクション種別ごとの典型的な変化量
        predictions = {
            "click": 0.02,        # 小さな変化（ボタン押下、フォーカス移動）
            "double_click": 0.05, # 中程度の変化（アプリ起動等）
            "type": 0.01,         # テキスト表示
            "keypress": 0.03,     # 状態変化
            "scroll": 0.08,       # スクロールで画面全体が変化
            "focus_window": 0.15, # ウィンドウ切替で大きな変化
            "wait": 0.0,          # 変化なし
            "none": 0.0,
        }
        self._expected_diff = predictions.get(action_type, 0.02)
        return self._expected_diff

    def evaluate(self, actual_diff: float) -> dict:
        """
        予測と実際の差分を比較し、予測誤差を計算する。

        Returns
        -------
        dict
            prediction_error: 正規化された予測誤差（0=完全一致）
            is_surprising: 閾値を超えたか
            assessment: テキスト評価
        """
        if self._expected_diff == 0 and actual_diff == 0:
            error = 0.0
        elif self._expected_diff == 0:
            error = actual_diff  # 予測ゼロなのに変化あり
        else:
            error = abs(actual_diff - self._expected_diff) / max(self._expected_diff, 0.001)

        is_surprising = error > 2.0  # 予測の2倍以上のずれ

        if is_surprising:
            assessment = (f"⚠ 予測誤差: 予想={self._expected_diff:.3f}, "
                          f"実際={actual_diff:.3f}, 誤差={error:.1f}倍")
        else:
            assessment = f"予測通り: 予想={self._expected_diff:.3f}, 実際={actual_diff:.3f}"

        record = {
            "expected": self._expected_diff,
            "actual": actual_diff,
            "error": error,
            "surprising": is_surprising,
            "timestamp": time.time(),
        }
        self._history.append(record)
        # 履歴は直近50件まで
        if len(self._history) > 50:
            self._history = self._history[-50:]

        return {
            "prediction_error": error,
            "is_surprising": is_surprising,
            "assessment": assessment,
        }


# ===========================================================================
# Phase 3: 思考深度の動的調整
# ===========================================================================

class ThinkingDepthController:
    """
    タスク複雑度と確信度の変化率から、System 2 の思考深度（最大反復回数）を
    動的に調整する。TH2T（Think-How-to-Think）の知見に基づく。

    浅い思考（1-2反復）: 高確信度、単純タスク
    中程度（3反復）: 標準
    深い思考（4-5反復）: 低確信度、複雑タスク、連続失敗
    """

    MIN_DEPTH = 1
    MAX_DEPTH = 5
    BASE_DEPTH = 3

    def calculate_depth(self, ctx: GatekeeperContext,
                        confidence_history: List[float] = None) -> int:
        """
        コンテキストから最適な思考深度を計算する。

        Parameters
        ----------
        ctx : GatekeeperContext
            ゲートキーパーのコンテキスト
        confidence_history : list of float, optional
            直近の確信度の履歴（変化率の計算に使用）

        Returns
        -------
        int
            推奨される最大反復回数（1-5）
        """
        depth = self.BASE_DEPTH

        # 低確信度 → 深く考える
        if ctx.confidence < 0.5:
            depth += 2
        elif ctx.confidence < 0.7:
            depth += 1

        # 連続失敗 → 深く考える
        depth += min(ctx.consecutive_failures, 2)

        # 複雑なタスク → 深く考える
        if ctx.estimated_steps >= 5:
            depth += 1

        # 確信度が改善傾向 → 浅くてよい
        if confidence_history and len(confidence_history) >= 2:
            trend = confidence_history[-1] - confidence_history[-2]
            if trend > 0.1:  # 改善傾向
                depth -= 1
            elif trend < -0.1:  # 悪化傾向
                depth += 1

        # リスクの高い操作 → 深く考える
        if ctx.is_risky_action:
            depth += 1

        return max(self.MIN_DEPTH, min(self.MAX_DEPTH, depth))


# ===========================================================================
# Phase 3: パフォーマンス計測と閾値チューニング
# ===========================================================================

class PerformanceTracker:
    """
    System 1/2 の使用率、成功率、レイテンシを追跡し、
    ゲートキーパーの閾値を自動調整する。

    定期的に統計を分析し、System 2 が過剰に起動されている場合は
    閾値を緩和し、System 1 での失敗が多い場合は閾値を厳格化する。
    """

    TUNING_WINDOW = 20  # 直近N件で閾値を調整
    MIN_CONFIDENCE_THRESHOLD = 0.5
    MAX_CONFIDENCE_THRESHOLD = 0.9

    def __init__(self):
        self._records: List[dict] = []

    def record(self, mode: ThinkingMode, success: bool,
               confidence: float, duration: float = 0.0) -> None:
        """1ステップの結果を記録する。"""
        self._records.append({
            "mode": mode.value,
            "success": success,
            "confidence": confidence,
            "duration": duration,
            "timestamp": time.time(),
        })
        # 直近200件まで保持
        if len(self._records) > 200:
            self._records = self._records[-200:]

    def get_stats(self) -> dict:
        """現在の統計情報を返す。"""
        if not self._records:
            return {"total": 0}

        s1 = [r for r in self._records if r["mode"] == "system1"]
        s2 = [r for r in self._records if r["mode"] == "system2"]

        s1_success = sum(1 for r in s1 if r["success"]) / max(len(s1), 1)
        s2_success = sum(1 for r in s2 if r["success"]) / max(len(s2), 1)
        s2_avg_dur = (sum(r["duration"] for r in s2) / max(len(s2), 1))

        return {
            "total": len(self._records),
            "system1_count": len(s1),
            "system2_count": len(s2),
            "system1_success_rate": round(s1_success, 2),
            "system2_success_rate": round(s2_success, 2),
            "system2_ratio": round(len(s2) / max(len(self._records), 1), 2),
            "system2_avg_duration": round(s2_avg_dur, 2),
        }

    def suggest_threshold(self, current_threshold: float) -> float:
        """
        直近の実績から、ゲートキーパーの確信度閾値を提案する。

        - System 1 の失敗率が高い → 閾値を上げる（System 2 をもっと使う）
        - System 2 の使用率が高すぎる → 閾値を下げる（System 1 に任せる）
        """
        recent = self._records[-self.TUNING_WINDOW:]
        if len(recent) < self.TUNING_WINDOW:
            return current_threshold  # データ不足

        s1_recent = [r for r in recent if r["mode"] == "system1"]
        s2_recent = [r for r in recent if r["mode"] == "system2"]

        s1_failures = sum(1 for r in s1_recent if not r["success"])
        s1_failure_rate = s1_failures / max(len(s1_recent), 1)
        s2_ratio = len(s2_recent) / len(recent)

        new_threshold = current_threshold

        # System 1 の失敗率が30%超 → 閾値を上げる（もっと慎重に）
        if s1_failure_rate > 0.3:
            new_threshold = min(current_threshold + 0.05,
                                self.MAX_CONFIDENCE_THRESHOLD)

        # System 2 の使用率が60%超 → 閾値を下げる（もっと信頼して）
        elif s2_ratio > 0.6:
            new_threshold = max(current_threshold - 0.05,
                                self.MIN_CONFIDENCE_THRESHOLD)

        return round(new_threshold, 2)
