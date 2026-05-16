"""
test_agent.py
Video2AI エージェントモジュールの単体テスト・統合テスト

ドライランモードでエージェントループ全体の動作を検証する。
LLM APIへの実際の呼び出しはモックで代替。
"""
import os
import sys
import time
import threading
import json
import numpy as np
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# video2aiルートをパスに追加
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ["DISPLAY"] = ":99"

from agent.change_detector import ChangeDetector
from agent.action_executor import ActionExecutor
from agent.session_logger import SessionLogger


class TestChangeDetector(unittest.TestCase):
    """ChangeDetectorの単体テスト。"""

    def _make_frame(self, brightness: int = 128, size=(100, 100)) -> np.ndarray:
        """テスト用フレームを生成する。"""
        return np.full((size[0], size[1], 3), brightness, dtype=np.uint8)

    def test_first_frame_returns_zero(self):
        """最初のフレームは変化スコア0を返す。"""
        detector = ChangeDetector(method="diff")
        frame = self._make_frame(128)
        score = detector.compute_score(frame)
        self.assertEqual(score, 0.0)

    def test_identical_frames_no_change(self):
        """同一フレームは変化なしと判定される。"""
        detector = ChangeDetector(method="diff", threshold=0.02)
        frame = self._make_frame(128)
        detector.compute_score(frame)  # 初期化
        changed, score = detector.is_changed(frame)
        self.assertFalse(changed)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_different_frames_change_detected(self):
        """大きく異なるフレームは変化ありと判定される。"""
        detector = ChangeDetector(method="diff", threshold=0.02)
        frame1 = self._make_frame(0)    # 黒
        frame2 = self._make_frame(255)  # 白
        detector.compute_score(frame1)  # 初期化
        changed, score = detector.is_changed(frame2)
        self.assertTrue(changed)
        self.assertGreater(score, 0.5)

    def test_ssim_method(self):
        """SSIMメソッドが正常に動作する。"""
        detector = ChangeDetector(method="ssim", threshold=0.05)
        frame1 = self._make_frame(100)
        frame2 = self._make_frame(200)
        detector.compute_score(frame1)
        changed, score = detector.is_changed(frame2)
        self.assertTrue(changed)

    def test_optical_flow_method(self):
        """Optical Flowメソッドが正常に動作する。"""
        detector = ChangeDetector(method="optical-flow", threshold=0.01)
        frame1 = self._make_frame(100)
        frame2 = self._make_frame(100)
        # 右半分を変える（動き）
        frame2[:, 50:, :] = 200
        detector.compute_score(frame1)
        score = detector.compute_score(frame2)
        self.assertGreaterEqual(score, 0.0)

    def test_reset(self):
        """resetで状態がリセットされる。"""
        detector = ChangeDetector(method="diff")
        frame = self._make_frame(128)
        detector.compute_score(frame)
        detector.reset()
        score = detector.compute_score(frame)
        self.assertEqual(score, 0.0)

    def test_stats(self):
        """統計情報が正しく返される。"""
        detector = ChangeDetector(method="diff", threshold=0.02)
        frame1 = self._make_frame(0)
        frame2 = self._make_frame(255)
        detector.compute_score(frame1)
        detector.is_changed(frame2)
        stats = detector.stats
        self.assertEqual(stats["method"], "diff")
        self.assertEqual(stats["frame_count"], 1)
        self.assertEqual(stats["change_count"], 1)

    def test_default_thresholds(self):
        """各手法のデフォルト閾値が設定される。"""
        for method, expected in [("diff", 0.02), ("ssim", 0.05), ("optical-flow", 0.03)]:
            detector = ChangeDetector(method=method)
            self.assertEqual(detector.threshold, expected)


class TestActionExecutor(unittest.TestCase):
    """ActionExecutorの単体テスト（ドライランモード）。"""

    def setUp(self):
        self.executor = ActionExecutor(dry_run=True, display=":99")

    def test_click_dry_run(self):
        """ドライランでクリックが成功を返す。"""
        result = self.executor.click(100, 200)
        self.assertTrue(result.success)
        self.assertTrue(result.dry_run)
        self.assertIn("DRY-RUN", result.description)

    def test_type_dry_run(self):
        """ドライランでテキスト入力が成功を返す。"""
        result = self.executor.type_text("Hello, World!")
        self.assertTrue(result.success)
        self.assertTrue(result.dry_run)

    def test_keypress_dry_run(self):
        """ドライランでキー入力が成功を返す。"""
        result = self.executor.keypress("ctrl+s")
        self.assertTrue(result.success)
        self.assertTrue(result.dry_run)

    def test_scroll_dry_run(self):
        """ドライランでスクロールが成功を返す。"""
        result = self.executor.scroll(500, 400, "down", 3)
        self.assertTrue(result.success)
        self.assertTrue(result.dry_run)

    def test_wait_dry_run(self):
        """ドライランで待機が成功を返す。"""
        result = self.executor.wait(0.1)
        self.assertTrue(result.success)
        self.assertTrue(result.dry_run)

    def test_execute_suggestion_click(self):
        """ActionSuggestionのclickが正しく実行される。"""
        from agent.llm_vision import ActionSuggestion
        suggestion = ActionSuggestion(
            action_type="click",
            target="送信ボタン",
            value="",
            x=300,
            y=400,
            confidence=0.9,
            reasoning="送信ボタンが見えるため",
            analysis="フォーム画面",
        )
        result = self.executor.execute_suggestion(suggestion)
        self.assertTrue(result.success)

    def test_execute_suggestion_none(self):
        """noneアクションが正しく処理される。"""
        from agent.llm_vision import ActionSuggestion
        suggestion = ActionSuggestion(
            action_type="none",
            target="",
            value="",
            x=None,
            y=None,
            confidence=0.8,
            reasoning="変化待ち",
            analysis="待機中",
        )
        result = self.executor.execute_suggestion(suggestion)
        self.assertTrue(result.success)
        self.assertEqual(result.action_type, "none")


class TestSessionLogger(unittest.TestCase):
    """SessionLoggerの単体テスト。"""

    def setUp(self):
        self.test_dir = "/tmp/test_video2ai_agent"
        self.logger = SessionLogger(
            session_id="test-001",
            output_dir=self.test_dir,
            goal="テスト目標",
            model="test-model",
            method="diff",
            threshold=0.02,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_log_frame(self):
        """フレームログが正しく記録される。"""
        log = self.logger.log_frame(
            diff_score=0.05,
            changed=True,
            description="テスト画面",
            action_type="click",
            action_x=100,
            action_y=200,
        )
        self.assertEqual(log.diff_score, 0.05)
        self.assertTrue(log.changed)
        self.assertEqual(log.action_type, "click")
        self.assertEqual(log.frame_index, 0)

    def test_stats(self):
        """統計情報が正しく更新される。"""
        self.logger.log_frame(diff_score=0.01, changed=False, action_type="none")
        self.logger.log_frame(diff_score=0.05, changed=True, action_type="click")
        stats = self.logger.stats
        self.assertEqual(stats["frame_count"], 2)
        self.assertEqual(stats["changed_count"], 1)
        self.assertEqual(stats["action_counts"]["click"], 1)

    def test_end_session(self):
        """セッション終了時にサマリーが生成される。"""
        self.logger.log_frame(diff_score=0.05, changed=True, action_type="click")
        summary = self.logger.end_session()
        self.assertEqual(summary.session_id, "test-001")
        self.assertEqual(summary.total_frames, 1)
        self.assertEqual(summary.changed_frames, 1)
        self.assertEqual(summary.actions_taken, 1)
        self.assertGreater(summary.duration_seconds, 0)

    def test_jsonl_output(self):
        """JSONLファイルが正しく生成される。"""
        self.logger.log_frame(diff_score=0.03, changed=True, description="テスト")
        self.logger.end_session()
        log_path = Path(self.test_dir) / "test-001" / "events.jsonl"
        self.assertTrue(log_path.exists())
        lines = log_path.read_text().strip().split("\n")
        self.assertGreaterEqual(len(lines), 3)  # session_start + frame + session_end
        for line in lines:
            data = json.loads(line)
            self.assertIn("type", data)
            # session_startはstarted_at、frameはtimestampを持つ
            has_time = "timestamp" in data or "started_at" in data or "ended_at" in data
            self.assertTrue(has_time, f"time field missing in: {data}")

    def test_get_recent_logs(self):
        """直近ログが正しく返される。"""
        for i in range(5):
            self.logger.log_frame(diff_score=0.01 * i, changed=i > 2)
        recent = self.logger.get_recent_logs(3)
        self.assertEqual(len(recent), 3)


class TestAgentLoopDryRun(unittest.TestCase):
    """エージェントループの統合テスト（ドライランモード、LLMモック）。"""

    @patch("agent.llm_vision.LLMVision")
    def test_dry_run_loop(self, MockVision):
        """ドライランで数ステップ実行して正常終了する。"""
        from agent.llm_vision import ActionSuggestion, VisionAnalysis

        # LLM Visionのモック
        mock_vision = MagicMock()
        mock_vision.analyze.return_value = VisionAnalysis(
            description="テスト画面が表示されています",
            changes="画面が変化しました",
            suggested_action=ActionSuggestion(
                action_type="none",
                target="",
                value="",
                x=None,
                y=None,
                confidence=0.8,
                reasoning="監視継続",
                analysis="テスト",
            ),
        )
        MockVision.return_value = mock_vision

        events = []
        def on_event(event_type, data):
            events.append({"type": event_type, **data})

        from agent_loop import run_agent_loop
        summary = run_agent_loop(
            goal="テスト目標",
            method="diff",
            threshold=0.01,
            interval=0.05,
            model="test-model",
            max_steps=3,
            dry_run=True,
            display=":99",
            output_dir="/tmp/test_agent_loop",
            verbose=False,
            on_event=on_event,
        )

        self.assertIn("session_id", summary)
        self.assertEqual(summary["total_frames"], 3)
        self.assertGreaterEqual(summary["duration_seconds"], 0)

        # イベントが正しく発行されているか確認
        event_types = [e["type"] for e in events]
        self.assertIn("session_start", event_types)
        self.assertIn("session_end", event_types)

        import shutil
        shutil.rmtree("/tmp/test_agent_loop", ignore_errors=True)


class TestSafetyRules(unittest.TestCase):
    """安全ルールチェッカーのテスト。"""

    def _make_suggestion(self, action_type="click", x=500, y=300, value="", target="", reasoning=""):
        """テスト用のActionSuggestion風オブジェクトを作成する。"""
        class FakeSuggestion:
            pass
        s = FakeSuggestion()
        s.action_type = action_type
        s.x = x
        s.y = y
        s.value = value
        s.target = target
        s.reasoning = reasoning
        s.confidence = 0.9
        return s

    def test_no_rules_allows_all(self):
        from agent.safety_rules import SafetyRuleChecker, SafetyDecision
        checker = SafetyRuleChecker()
        verdict = checker.check(self._make_suggestion())
        self.assertEqual(verdict.decision, SafetyDecision.ALLOW)

    def test_forbidden_region_denies(self):
        import json, tempfile
        from agent.safety_rules import SafetyRuleChecker, SafetyDecision
        rules = {"rules": [
            {"id": "block-top", "type": "forbidden_region",
             "region": {"x1": 0, "y1": 0, "x2": 100, "y2": 50},
             "message": "Top-left blocked"}
        ]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(rules, f)
            path = f.name
        try:
            checker = SafetyRuleChecker(path)
            # Inside forbidden region
            verdict = checker.check(self._make_suggestion(x=50, y=25))
            self.assertEqual(verdict.decision, SafetyDecision.DENY)
            # Outside
            verdict = checker.check(self._make_suggestion(x=200, y=200))
            self.assertEqual(verdict.decision, SafetyDecision.ALLOW)
        finally:
            os.unlink(path)

    def test_forbidden_key_denies(self):
        import json, tempfile
        from agent.safety_rules import SafetyRuleChecker, SafetyDecision
        rules = {"rules": [
            {"id": "no-alt-f4", "type": "forbidden_key",
             "keys": ["alt+F4", "ctrl+alt+delete"],
             "message": "Dangerous key blocked"}
        ]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(rules, f)
            path = f.name
        try:
            checker = SafetyRuleChecker(path)
            verdict = checker.check(self._make_suggestion(action_type="keypress", value="alt+f4"))
            self.assertEqual(verdict.decision, SafetyDecision.DENY)
            verdict = checker.check(self._make_suggestion(action_type="keypress", value="ctrl+s"))
            self.assertEqual(verdict.decision, SafetyDecision.ALLOW)
        finally:
            os.unlink(path)

    def test_keyword_confirm(self):
        import json, tempfile
        from agent.safety_rules import SafetyRuleChecker, SafetyDecision
        rules = {"rules": [
            {"id": "confirm-delete", "type": "keyword_confirm",
             "keywords": ["delete", "remove"],
             "fields": ["target", "reasoning"],
             "message": "Destructive action"}
        ]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(rules, f)
            path = f.name
        try:
            checker = SafetyRuleChecker(path)
            verdict = checker.check(self._make_suggestion(target="Delete button"))
            self.assertEqual(verdict.decision, SafetyDecision.CONFIRM)
            verdict = checker.check(self._make_suggestion(target="Save button"))
            self.assertEqual(verdict.decision, SafetyDecision.ALLOW)
        finally:
            os.unlink(path)

    def test_deny_overrides_confirm(self):
        import json, tempfile
        from agent.safety_rules import SafetyRuleChecker, SafetyDecision
        rules = {"rules": [
            {"id": "confirm-region", "type": "region_confirm",
             "region": {"x1": 0, "y1": 0, "x2": 200, "y2": 200},
             "message": "Region needs confirm"},
            {"id": "block-region", "type": "forbidden_region",
             "region": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
             "message": "Region blocked"},
        ]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(rules, f)
            path = f.name
        try:
            checker = SafetyRuleChecker(path)
            # In both regions: DENY should win over CONFIRM
            verdict = checker.check(self._make_suggestion(x=50, y=50))
            self.assertEqual(verdict.decision, SafetyDecision.DENY)
        finally:
            os.unlink(path)

    def test_non_click_action_ignores_region_rules(self):
        import json, tempfile
        from agent.safety_rules import SafetyRuleChecker, SafetyDecision
        rules = {"rules": [
            {"id": "block-all", "type": "forbidden_region",
             "region": {"x1": 0, "y1": 0, "x2": 9999, "y2": 9999},
             "message": "All blocked"},
        ]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(rules, f)
            path = f.name
        try:
            checker = SafetyRuleChecker(path)
            # keypress should not be affected by region rules
            verdict = checker.check(self._make_suggestion(action_type="keypress", value="enter"))
            self.assertEqual(verdict.decision, SafetyDecision.ALLOW)
        finally:
            os.unlink(path)


class TestActionPreviewCancelled(unittest.TestCase):
    """ActionResult.cancelled フィールドのテスト。"""

    def test_cancelled_field_exists(self):
        from agent.action_executor import ActionResult
        result = ActionResult(False, "click", "test", cancelled=True)
        self.assertTrue(result.cancelled)

    def test_default_not_cancelled(self):
        from agent.action_executor import ActionResult
        result = ActionResult(True, "click", "test")
        self.assertFalse(result.cancelled)


class TestParallelCoordinator(unittest.TestCase):
    """並列ワーカー/コーディネーターのテスト。"""

    def test_worker_report_creation(self):
        from agent.parallel_coordinator import WorkerReport
        report = WorkerReport(
            frame_index=1, captured_at=1000.0, reported_at=1003.0,
            description="テスト画面", changes="ボタンが表示された",
            diff_score=0.05, worker_id=0,
        )
        self.assertEqual(report.frame_index, 1)
        self.assertEqual(report.description, "テスト画面")
        self.assertEqual(report.worker_id, 0)

    def test_frame_dispatcher_enqueues_reports(self):
        import queue as q
        from unittest.mock import MagicMock
        from agent.parallel_coordinator import FrameDispatcher

        mock_vision = MagicMock()
        mock_vision.analyze_observation_only.return_value = {
            "description": "テスト画面", "changes": "変化あり",
        }
        report_queue = q.Queue()
        dispatcher = FrameDispatcher(
            num_workers=2, vision_worker=mock_vision,
            report_queue=report_queue, screen_size=(1920, 1080),
        )
        # 3フレームをディスパッチ
        for i in range(3):
            dispatcher.dispatch(f"base64_{i}", i, 0.05, time.time())
        time.sleep(1.0)  # ワーカーの処理を待つ
        dispatcher.shutdown()

        reports = []
        while not report_queue.empty():
            reports.append(report_queue.get())
        self.assertEqual(len(reports), 3)
        self.assertEqual(reports[0].description, "テスト画面")

    def test_coordinator_filters_stale_reports(self):
        import queue as q
        from unittest.mock import MagicMock
        from agent.parallel_coordinator import ActionCoordinator, WorkerReport

        report_queue = q.Queue()
        mock_vision = MagicMock()
        mock_executor = MagicMock()
        mock_logger = MagicMock()

        coordinator = ActionCoordinator(
            report_queue=report_queue,
            vision_coordinator=mock_vision,
            executor=mock_executor,
            safety_checker=None,
            logger=mock_logger,
            goal="test",
            staleness_threshold=2.0,
        )
        # 古い報告を追加（10秒前）
        report_queue.put(WorkerReport(
            frame_index=1, captured_at=time.time() - 10, reported_at=time.time(),
            description="古い", changes="", diff_score=0.1, worker_id=0,
        ))
        coordinator._process_reports()
        # 古いので処理されない → LLM呼び出しなし
        mock_vision.decide_action.assert_not_called()

    def test_coordinator_processes_fresh_reports(self):
        import queue as q
        from unittest.mock import MagicMock
        from agent.parallel_coordinator import ActionCoordinator, WorkerReport
        from agent.llm_vision import ActionSuggestion

        report_queue = q.Queue()
        mock_vision = MagicMock()
        mock_vision.decide_action.return_value = ActionSuggestion(
            action_type="none", target="", value="", x=None, y=None,
            confidence=0.3, reasoning="待機", analysis="",
        )
        mock_executor = MagicMock()
        mock_logger = MagicMock()

        coordinator = ActionCoordinator(
            report_queue=report_queue,
            vision_coordinator=mock_vision,
            executor=mock_executor,
            safety_checker=None,
            logger=mock_logger,
            goal="test",
            staleness_threshold=5.0,
        )
        # 新しい報告を追加
        report_queue.put(WorkerReport(
            frame_index=1, captured_at=time.time(), reported_at=time.time(),
            description="新しい画面", changes="変化", diff_score=0.05, worker_id=0,
        ))
        coordinator._process_reports()
        # 新しいのでLLM呼び出しあり
        mock_vision.decide_action.assert_called_once()

    def test_session_logger_thread_safety(self):
        """複数スレッドから同時にlog_frameを呼んでもクラッシュしない。"""
        import tempfile
        from agent.session_logger import SessionLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(session_id="test-thread", output_dir=tmpdir)
            errors = []

            def log_many(start):
                try:
                    for i in range(20):
                        logger.log_frame(diff_score=0.01, changed=False, frame_index=start + i)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=log_many, args=(i * 20,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(len(errors), 0)
            self.assertEqual(logger._frame_count, 80)


if __name__ == "__main__":
    unittest.main(verbosity=2)
