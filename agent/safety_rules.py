"""
agent/safety_rules.py
アプリケーション別安全ルールチェッカー

JSONで定義されたルールに基づき、ActionSuggestion の実行前に
DENY（拒否）/ CONFIRM（確認）/ ALLOW（許可）を判定する。

ルール種別:
  forbidden_region  : 指定座標範囲へのクリック/スクロールを拒否
  forbidden_key     : 指定キーコンボを拒否
  keyword_confirm   : LLMの説明に特定キーワードがあれば確認を要求
  region_confirm    : 指定座標範囲へのクリック時に確認を要求

使い方:
    checker = SafetyRuleChecker("safety_rules.json")
    verdict = checker.check(suggestion)
    if verdict.decision == SafetyDecision.DENY:
        print(f"Blocked: {verdict.message}")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class SafetyDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    CONFIRM = "confirm"


@dataclass
class SafetyVerdict:
    decision: SafetyDecision
    matched_rule_id: str = ""
    message: str = ""


def _in_region(x: Optional[int], y: Optional[int], region: dict) -> bool:
    """座標が矩形領域内にあるか判定する。"""
    if x is None or y is None:
        return False
    return (region.get("x1", 0) <= x <= region.get("x2", 0) and
            region.get("y1", 0) <= y <= region.get("y2", 0))


# ユーザーの作業環境を破壊する危険なキー操作
_BUILTIN_FORBIDDEN_KEYS = {
    "win+d": "全ウィンドウをデスクトップ表示（作業破壊）",
    "win+m": "全ウィンドウを最小化（作業破壊）",
    "alt+f4": "ウィンドウを閉じる（意図しない終了）",
    "ctrl+w": "タブ/ウィンドウを閉じる（意図しない終了）",
    "ctrl+alt+delete": "システムメニュー（操作中断）",
}


class SafetyRuleChecker:
    """ActionSuggestion を安全ルールに照合して判定を返す。"""

    def __init__(self, rules_path: Optional[str] = None):
        self.rules: list[dict] = []
        if rules_path:
            path = Path(rules_path)
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                self.rules = data.get("rules", [])

    def check(self, suggestion) -> SafetyVerdict:
        """
        全ルールを評価し、最も厳しい判定を返す。
        優先順位: DENY > CONFIRM > ALLOW
        """
        # 組み込み禁止キーチェック
        action = getattr(suggestion, "action_type", "").lower()
        if action == "keypress":
            value = getattr(suggestion, "value", "").lower().replace(" ", "")
            for forbidden, reason in _BUILTIN_FORBIDDEN_KEYS.items():
                if value == forbidden.replace(" ", ""):
                    return SafetyVerdict(SafetyDecision.DENY, f"builtin_{forbidden}", f"禁止キー: {reason}")

        denies: list[SafetyVerdict] = []
        confirms: list[SafetyVerdict] = []

        for rule in self.rules:
            verdict = self._evaluate_rule(rule, suggestion)
            if verdict is None:
                continue
            if verdict.decision == SafetyDecision.DENY:
                denies.append(verdict)
            elif verdict.decision == SafetyDecision.CONFIRM:
                confirms.append(verdict)

        if denies:
            return denies[0]
        if confirms:
            return confirms[0]
        return SafetyVerdict(SafetyDecision.ALLOW)

    def _evaluate_rule(self, rule: dict, suggestion) -> Optional[SafetyVerdict]:
        rule_type = rule.get("type", "")
        rule_id = rule.get("id", "unknown")
        message = rule.get("message", rule_type)

        if rule_type == "forbidden_region":
            return self._check_forbidden_region(rule, suggestion, rule_id, message)
        elif rule_type == "forbidden_key":
            return self._check_forbidden_key(rule, suggestion, rule_id, message)
        elif rule_type == "keyword_confirm":
            return self._check_keyword_confirm(rule, suggestion, rule_id, message)
        elif rule_type == "region_confirm":
            return self._check_region_confirm(rule, suggestion, rule_id, message)
        return None

    def _check_forbidden_region(self, rule, suggestion, rule_id, message) -> Optional[SafetyVerdict]:
        action = getattr(suggestion, "action_type", "").lower()
        if action not in ("click", "double_click", "scroll"):
            return None
        region = rule.get("region", {})
        x = getattr(suggestion, "x", None)
        y = getattr(suggestion, "y", None)
        if _in_region(x, y, region):
            return SafetyVerdict(SafetyDecision.DENY, rule_id, message)
        return None

    def _check_forbidden_key(self, rule, suggestion, rule_id, message) -> Optional[SafetyVerdict]:
        action = getattr(suggestion, "action_type", "").lower()
        if action != "keypress":
            return None
        value = getattr(suggestion, "value", "").lower()
        forbidden = [k.lower() for k in rule.get("keys", [])]
        if value in forbidden:
            return SafetyVerdict(SafetyDecision.DENY, rule_id, message)
        return None

    def _check_keyword_confirm(self, rule, suggestion, rule_id, message) -> Optional[SafetyVerdict]:
        keywords = [k.lower() for k in rule.get("keywords", [])]
        fields = rule.get("fields", ["target", "value", "reasoning"])
        for field in fields:
            text = str(getattr(suggestion, field, "")).lower()
            for kw in keywords:
                if kw in text:
                    return SafetyVerdict(SafetyDecision.CONFIRM, rule_id, f"{message} ('{kw}' in {field})")
        return None

    def _check_region_confirm(self, rule, suggestion, rule_id, message) -> Optional[SafetyVerdict]:
        action = getattr(suggestion, "action_type", "").lower()
        if action not in ("click", "double_click", "scroll"):
            return None
        region = rule.get("region", {})
        x = getattr(suggestion, "x", None)
        y = getattr(suggestion, "y", None)
        if _in_region(x, y, region):
            return SafetyVerdict(SafetyDecision.CONFIRM, rule_id, message)
        return None
