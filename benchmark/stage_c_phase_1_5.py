"""Phase 1.5 heuristic declare_target helper.

Commit 3 (Option iii, supervisor dialogue pending — see
dlg/cognitive-nodes/..T19:12:16-15ee): ships a deterministic rule-based
target-type declarer instead of a real Claude tool-use call. Rationale in
the submission post; short version:

- Heuristic routing has zero variance, so an A/B smoke (Stage B vs Stage C
  via heuristic) isolates Stage C's CV contribution from any VLM sampling
  variance. That matches decision 050 論点 1's *spirit* (contract clarity
  + variance locality) more cleanly than a tool-use-sourced target_type
  ever could.
- Real Claude tool-use stays on the roadmap as a follow-up commit; the
  `declare_target` contract here is the same one a tool-use call would
  emit, so the swap is mechanical.

Contract:

    declare_target(target_prompt: str) -> dict {
        "target_type": str,     # one of stage_c.SUPPORTED_TARGET_TYPES
        "intent":      str,     # "center" / "corner" / "edge" / "tip" /
                                #   "intersection" / "top" / "bottom"
        "label_hint":  str,     # extracted quoted label, or ""
        "source":      str,     # "heuristic" for this implementation;
                                # real tool-use will emit "vlm_tool_use".
                                # Promoted to top-level `stage_c_source` by
                                # the caller (decision 052 9-field canonical).
        "reasoning":   str,     # rule hit id, identifying which heuristic
                                # branch fired. One of:
                                #   "button_label_match" / "icon_label_match" /
                                #   "menu_label_match" / "label_only_fallback" /
                                #   "no_rule_fired" / "empty_prompt".
                                # Promoted to top-level `stage_c_reasoning` by
                                # the caller. For a future tool-use variant,
                                # this field carries the VLM's raw tool input
                                # instead.
        "visible":     bool,    # always True for heuristic — visibility
                                # is still checked downstream by Phase 1.5
                                # anchor declaration (unchanged)
    }

The 50-case ground_truth.json splits cleanly by these rules:

    ui_button × 10   → target_type=button    (all contain "button")
    ui_dense  × 10   → target_type=icon      (all contain "icon")
    natural   × 10   → target_type=generic   (no quoted label, descriptive)
    ambiguous × 10   → target_type=generic
    precision × 10   → target_type=generic

So Stage C routes roughly 20/50 targets to a specialized pipeline
(button/icon) and 30/50 to the measurement-honest generic miss path.
"""
from __future__ import annotations

import json as _json
import logging as _logging
import re as _re
import time as _time
from typing import Any, Dict, Optional

from benchmark import stage_b as _stage_b
from benchmark import stage_c as _stage_c

_BUTTON_PAT = _re.compile(r"\bbutton\b", _re.IGNORECASE)
_ICON_PAT = _re.compile(r"\bicon\b", _re.IGNORECASE)
_MENU_PAT = _re.compile(r"\bmenu(\s*item)?\b", _re.IGNORECASE)

# Intent keywords that appear in benchmark prompts. The current contour
# pipelines ignore intent (Commit 2 scope), but we still record it so future
# commits can branch (e.g. "tip" → edge extremum rather than centroid) and
# so the paper can report per-intent breakdowns.
_INTENT_ORDER = (
    ("intersection", _re.compile(r"\bintersection\b", _re.IGNORECASE)),
    ("tip", _re.compile(r"\btip\b", _re.IGNORECASE)),
    ("top", _re.compile(r"\btop\b", _re.IGNORECASE)),
    ("bottom", _re.compile(r"\bbottom\b", _re.IGNORECASE)),
    ("corner", _re.compile(r"\bcorner\b", _re.IGNORECASE)),
    ("edge", _re.compile(r"\bedge\b", _re.IGNORECASE)),
    ("center", _re.compile(r"\bcenter\b", _re.IGNORECASE)),
)


def declare_target(target_prompt: str) -> Dict[str, Any]:
    """Deterministic heuristic implementation of Phase 1.5 target declaration."""
    if not target_prompt:
        return {
            "target_type": "generic",
            "intent": "center",
            "label_hint": "",
            "source": "heuristic",
            "reasoning": "empty_prompt",
            "visible": True,
        }

    label = _stage_b.extract_target_label(target_prompt)

    if _BUTTON_PAT.search(target_prompt) and label:
        target_type = "button"
        reasoning = "button_label_match"
    elif _ICON_PAT.search(target_prompt) and label:
        target_type = "icon"
        reasoning = "icon_label_match"
    elif _MENU_PAT.search(target_prompt) and label:
        target_type = "menu_item"
        reasoning = "menu_label_match"
    else:
        target_type = "generic"
        # If a quoted label is present but no type keyword, use menu_item so
        # Stage B's OCR-snap still has a chance (the original v7 behavior).
        # This keeps the Stage C routing at least as permissive as Stage B.
        if label:
            target_type = "menu_item"
            reasoning = "label_only_fallback"
        else:
            reasoning = "no_rule_fired"

    intent = "center"
    for name, pat in _INTENT_ORDER:
        if pat.search(target_prompt):
            intent = name
            break

    return {
        "target_type": target_type,
        "intent": intent,
        "label_hint": label,
        "source": "heuristic",
        "reasoning": reasoning,
        "visible": True,
    }


# ---------------------------------------------------------------------------
# Phase C Priority 4.1 — LLM-routed structured output via claude CLI
# (decision 063 candidate, supersedes Priority 4's SDK transport)
# ---------------------------------------------------------------------------
#
# arm B/C path: the heuristic regex rules above are replaced by a claude
# CLI subprocess that emits a structured JSON declaration of the target.
#
# Mechanism note: this is **LLM-routed structured output** with prompt-
# engineered schema enforcement (post-parse validated), not true tool-use
# at the model layer. claude CLI exposes `--tools` as a built-in tools
# allowlist (Read/Grep/Bash/...), not a user-defined tool-schema passthrough,
# so the Anthropic Messages-API `tools=[...]` + `tool_choice=...` pattern is
# not available via CLI. We compensate with a four-layer mitigation (strict
# system prompt + response extraction + required-field validation + retry).
# Paper §8.5.5 narrative reflects this honest distinction: arm B/C
# exercises a weaker LLM-level guarantee than true tool-use.
#
# Transport: `claude -p --output-format json --model ... --no-session-
# persistence --setting-sources user`. OAuth subscription is enforced by
# stripping ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN from the child env,
# matching the `grid_core._vlm_call_cli` pattern used by the Phase 1 grid
# descent.
#
# Shape equivalence: the happy-path return dict has the same keys as the
# heuristic variant ({target_type, intent, label_hint, source, reasoning,
# visible}) so the run_benchmark caller does not branch on `source`. The
# failure-mode return dict uses `source = "vlm_tool_use_failed"` and
# `reasoning = "parse_fail_<kind>"` so rollups can derive arm_failed = True
# without a separate flag field (supervisor Q4-5 resolution).
#
# Retry policy: on ValueError, sleep 1s / 2s (exponential) and retry up
# to 2 times (3 total attempts) before emitting the failure sentinel. Per
# W3 v2 §6.3 Must-fix B, failure rows are NOT substituted with the
# heuristic output — downstream rollup excludes them from variance calc.
#
# Parse-or-fail distinction (supervisor Q5 resolution):
#   - **coerce**: JSON parseable but enum value unsupported → caller
#     clamps `target_type` / `intent` to the default, success path.
#   - **fail**: JSON itself unparseable or required field missing →
#     ValueError → caller retries / emits `vlm_tool_use_failed`.

_TOOL_NAME = "declare_target"
_TOOL_DESCRIPTION = (
    "Given a short natural-language target description, declare the "
    "structured properties of that target so a downstream geometric "
    "pipeline can route correctly."
)
# Intent enum matches _INTENT_ORDER above; kept in sync.
_TOOL_INTENTS = [
    "center", "corner", "edge", "tip", "intersection", "top", "bottom",
]

# Defaults. Model name is forwarded from the caller when available so
# run_benchmark's --model flag still governs which model arm B/C exercises.
_TOOL_DEFAULT_MODEL = "claude-sonnet-4-20250514"
_TOOL_MAX_TOKENS = 200
_TOOL_MAX_RETRIES = 3   # 1 initial + 2 retries (supervisor Q4-5 "retry ×2")

# Exponential backoff base. Kept as a module constant so tests can monkey-
# patch it to 0 for fast runs. Never shortened below 0 at runtime.
_TOOL_BACKOFF_BASE_S = 1.0

# Subprocess call budget. Matches `grid_core._vlm_call_cli` for operational
# parity; CLI-side retries for rate limits + auth are transparent within
# this budget.
_CLI_TIMEOUT_S = 60


def _build_tool_schema() -> Dict[str, Any]:
    """JSON schema for the `declare_target` structured output contract.

    Historical name kept for back-compat: this function previously built
    the `input_schema` of an Anthropic tool-use call. Priority 4.1 moved
    the transport to the claude CLI, which has no tool-schema passthrough;
    the schema is now embedded into the system prompt by
    :func:`_build_system_prompt`. This builder is retained because the
    TDD harness / future re-introduction of tool-use would re-use it.
    """
    return {
        "type": "object",
        "properties": {
            "target_type": {
                "type": "string",
                "enum": list(_stage_c.SUPPORTED_TARGET_TYPES),
                "description": (
                    "Geometric class of the target. Pick 'generic' when "
                    "none of the specialised classes fits."
                ),
            },
            "intent": {
                "type": "string",
                "enum": _TOOL_INTENTS,
                "description": (
                    "Which point of the target the caller wants: center of "
                    "mass, tip of a shape, corner, etc."
                ),
            },
            "label_hint": {
                "type": "string",
                "description": (
                    "Quoted text label that identifies the target (e.g. "
                    "'Save' for a Save button). Empty string if no label."
                ),
            },
            "visible": {
                "type": "boolean",
                "description": "True if the target is plausibly on-screen.",
            },
        },
        "required": ["target_type", "intent", "label_hint", "visible"],
    }


def _build_system_prompt() -> str:
    """System prompt embedding the declare_target schema as strict JSON.

    Replaces the Messages-API `tools=[...]` + `tool_choice=...` mechanism
    that the claude CLI does not expose. The LLM is instructed to emit a
    bare JSON object; downstream code parses + validates (four-layer
    mitigation). The system prompt lists the enum values verbatim so a
    strict-output LLM does not need to recall the schema independently.
    """
    target_types = ", ".join(_stage_c.SUPPORTED_TARGET_TYPES)
    intents = ", ".join(_TOOL_INTENTS)
    return (
        "You are a routing classifier for a coordinate-refinement "
        "pipeline.\n\n"
        "Respond with ONLY a JSON object matching the schema below. No "
        "markdown, no code fences, no commentary, no preamble, no trailing "
        "text. Your entire response must be a single valid JSON object.\n\n"
        "Schema:\n"
        "{\n"
        f'  "target_type": one of [{target_types}],\n'
        f'  "intent": one of [{intents}],\n'
        '  "label_hint": string (quoted label text, or "" if no label),\n'
        '  "visible": boolean (true if plausibly on screen)\n'
        "}\n\n"
        "Pick 'target_type=generic' when none of the specialised classes "
        "fits. Pick 'intent=center' when no specific sub-point is named."
    )


def _extract_json_object(text: str) -> str:
    """Extract the first balanced top-level JSON object from ``text``.

    Defensive parsing layer 2 (layer 1 = CLI envelope JSON). Tolerates
    markdown fences (```json ... ```) and trailing commentary by walking
    the string and tracking brace depth. Returns the substring (including
    the braces) or ``""`` if no balanced pair exists.

    Four-layer mitigation (supervisor Q2 approved):
      1. prompt strictness (see :func:`_build_system_prompt`)
      2. response extraction (this function)
      3. required-field validation (in :func:`_call_claude_cli_tool_use`)
      4. retry ×2 with exp backoff (in :func:`declare_target_tool_use`)
    """
    if not text:
        return ""
    stripped = text.strip()
    # Drop a leading markdown code fence (`` ```json `` / `` ``` ``) if
    # present, and the matching trailing fence.
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    if start < 0:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(stripped)):
        c = stripped[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return stripped[start:i + 1]
    return ""


def _call_claude_cli_tool_use(prompt: str, *,
                               model: Optional[str] = None,
                               max_tokens: int = _TOOL_MAX_TOKENS,
                               timeout_s: float = _CLI_TIMEOUT_S
                               ) -> Dict[str, Any]:
    """Low-level CLI-based LLM-routed structured-output call.

    Spawns ``claude -p --output-format json`` with a system prompt that
    embeds the declare_target schema, parses the two-layer JSON (CLI
    envelope → assistant's inner JSON), defensively extracts the inner
    object, validates required fields, and returns a dict with keys
    ``target_type`` / ``intent`` / ``label_hint`` / ``visible`` / ``raw``.

    Raises ``ValueError`` on any layer failure:
      - subprocess non-zero / timeout / FileNotFoundError
      - unparseable CLI envelope / empty ``result`` field
      - unparseable inner JSON / non-object inner value
      - missing required schema field

    Enum coercion (unsupported ``target_type`` / ``intent``) is the
    caller's responsibility (supervisor Q5: parseable + invalid-enum
    = coerce at caller; unparseable = fail here and retry).

    This function is intentionally thin and free of retry logic so tests
    can patch it with ``side_effect=ValueError(...)`` to exercise the
    retry loop in :func:`declare_target_tool_use`. Priority 4 / 4.1
    TDD harness patches this function by name via the alias below.
    """
    import os as _os
    import subprocess as _sp

    mdl = model or _TOOL_DEFAULT_MODEL
    # Map full model names to CLI aliases (same pattern as _vlm_call_cli).
    lowered = str(mdl).lower()
    if "sonnet" in lowered:
        cli_model = "sonnet"
    elif "opus" in lowered:
        cli_model = "opus"
    elif "haiku" in lowered:
        cli_model = "haiku"
    else:
        cli_model = mdl

    system_prompt = _build_system_prompt()

    # Force OAuth subscription: strip API-key env vars from the child so
    # the CLI never accidentally routes via API billing. Same hygiene as
    # grid_core._vlm_call_cli for Phase 1 grid descent.
    child_env = _os.environ.copy()
    child_env.pop("ANTHROPIC_API_KEY", None)
    child_env.pop("ANTHROPIC_AUTH_TOKEN", None)

    try:
        result = _sp.run(
            ["claude", "-p", prompt,
             "--append-system-prompt", system_prompt,
             "--output-format", "json",
             "--model", cli_model,
             "--no-session-persistence",
             "--setting-sources", "user"],
            capture_output=True, text=True, timeout=timeout_s,
            encoding="utf-8", errors="replace",
            env=child_env,
        )
    except _sp.TimeoutExpired as e:
        raise ValueError("timeout") from e
    except FileNotFoundError as e:
        raise ValueError("cli_not_found") from e
    except Exception as e:  # pragma: no cover — defensive
        raise ValueError(f"subprocess_error_{type(e).__name__}") from e

    if result.returncode != 0:
        stderr = (result.stderr or "")[:120].replace("\n", " ").strip()
        raise ValueError(f"cli_rc_{result.returncode}_{stderr}")

    stdout = (result.stdout or "").strip()
    if not stdout:
        raise ValueError("cli_empty_stdout")

    # Layer 1: CLI envelope JSON. `claude -p --output-format json` emits a
    # single-line (or pretty-printed) JSON object whose `result` field
    # carries the assistant's reply.
    try:
        envelope = _json.loads(stdout)
    except _json.JSONDecodeError as e:
        raise ValueError("malformed_envelope_json") from e

    assistant_text = ""
    if isinstance(envelope, dict):
        assistant_text = envelope.get("result") or ""
    if not isinstance(assistant_text, str) or not assistant_text.strip():
        raise ValueError("empty_result_field")

    # Layer 2: defensive extraction of the inner JSON object.
    inner_str = _extract_json_object(assistant_text)
    if not inner_str:
        raise ValueError("no_inner_json_object")

    # Layer 3 (parse): inner JSON to dict.
    try:
        tool_input = _json.loads(inner_str)
    except _json.JSONDecodeError as e:
        raise ValueError("malformed_inner_json") from e
    if not isinstance(tool_input, dict):
        raise ValueError("inner_not_object")

    # Layer 3 (required fields): enum validity is coerced at the caller
    # per supervisor Q5. Here we only verify presence of the four keys.
    for field in ("target_type", "intent", "label_hint", "visible"):
        if field not in tool_input:
            raise ValueError(f"missing_field_{field}")

    return {
        "target_type": tool_input.get("target_type"),
        "intent": tool_input.get("intent"),
        "label_hint": tool_input.get("label_hint", ""),
        "visible": tool_input.get("visible", True),
        "raw": _json.dumps(tool_input, ensure_ascii=False, sort_keys=True),
    }


# Alias for test-harness compatibility (supervisor Q1/Q4 ALIAS PATTERN
# APPROVED). The SDK-based implementation was removed in Priority 4.1
# (decision 063 candidate); this alias preserves
# `test_arm_c_aggregator.py`'s `patch.object(p1_5, "_call_anthropic_tool_use")`
# target without requiring the benchmark-agent-authored test file to be
# edited. The function name retains the legacy Anthropic SDK reference
# for mock-point stability; the implementation now uses the claude CLI
# subprocess. A cosmetic rename can ship as a follow-up cleanup commit
# post-Stage 1 if the alias bothers future readers.
_call_anthropic_tool_use = _call_claude_cli_tool_use


def declare_target_tool_use(target_prompt: str, *,
                             api_key: Optional[str] = None,
                             model: Optional[str] = None) -> Dict[str, Any]:
    """LLM-routed structured-output variant of :func:`declare_target`
    (decision 063 candidate, supersedes decision 062's SDK transport).

    Same 6-key return contract as the heuristic variant but with
    ``source="vlm_tool_use"`` on success. On retry-exhausted failure, emits
    ``source="vlm_tool_use_failed"`` + ``reasoning="parse_fail_<kind>"`` so
    downstream rollup can derive ``arm_failed = True`` without a separate
    flag field (supervisor Q4-5 resolution). The ``vlm_tool_use*`` enum
    values are kept verbatim for schema stability even though the mechanism
    is CLI-based LLM-routed structured output rather than true Anthropic
    tool-use (supervisor Q2 terminology note — paper narrative revises at
    §8.5.5, schema unchanged).

    ``api_key`` parameter is accepted for interface stability with the
    run_benchmark caller but ignored: the CLI subprocess inherits OAuth
    subscription auth and strips API-key env vars from its child (see
    :func:`_call_claude_cli_tool_use`).

    Enum coercion (supervisor Q5 resolution): if the LLM returns an
    unsupported ``target_type`` / ``intent``, the value is clamped to the
    default (``"generic"`` / ``"center"``) rather than treated as a parse
    failure. Unparseable JSON or missing required fields are distinct and
    drive the retry + failure-sentinel path.
    """
    del api_key  # accepted for interface stability, not used on CLI path.

    if not target_prompt:
        # Mirrors the heuristic empty-prompt branch but tagged with the
        # CLI-path provenance so the two arm paths emit distinguishable
        # source values even on the degenerate input.
        return {
            "target_type": "generic",
            "intent": "center",
            "label_hint": "",
            "source": "vlm_tool_use_failed",
            "reasoning": "parse_fail_empty_prompt",
            "visible": True,
        }

    last_err = "unknown"
    for attempt in range(_TOOL_MAX_RETRIES):
        try:
            # NB: call via the alias `_call_anthropic_tool_use` (= CLI impl)
            # so tests that patch the legacy name still intercept.
            resp = _call_anthropic_tool_use(target_prompt, model=model)

            target_type = resp.get("target_type") or "generic"
            if target_type not in _stage_c.SUPPORTED_TARGET_TYPES:
                target_type = "generic"
            intent = resp.get("intent") or "center"
            if intent not in _TOOL_INTENTS:
                intent = "center"
            label_hint = resp.get("label_hint") or ""
            raw = resp.get("raw") or ""
            if not raw:
                raw = _json.dumps(
                    {"target_type": target_type, "intent": intent,
                     "label_hint": label_hint}, sort_keys=True)

            visible_raw = resp.get("visible", True)
            visible = bool(visible_raw) if visible_raw is not None else True

            return {
                "target_type": target_type,
                "intent": intent,
                "label_hint": label_hint,
                "source": "vlm_tool_use",
                "reasoning": raw,
                "visible": visible,
            }
        except ValueError as e:
            msg = str(e) or "malformed_json"
            last_err = msg.split(":")[0].strip() or "malformed_json"
            _logging.warning(
                "declare_target_tool_use attempt %d/%d failed: %s",
                attempt + 1, _TOOL_MAX_RETRIES, msg)
        except Exception as e:  # pragma: no cover — defensive
            last_err = f"unexpected_{type(e).__name__}"
            _logging.warning(
                "declare_target_tool_use attempt %d/%d unexpected: %s",
                attempt + 1, _TOOL_MAX_RETRIES, e)

        if attempt < _TOOL_MAX_RETRIES - 1:
            _time.sleep(max(0.0, _TOOL_BACKOFF_BASE_S * (2 ** attempt)))

    # Retry exhausted — sentinel. `parse_fail_<kind>` prefix lets rollup code
    # pattern-match on kind for triage without parsing the enum value itself.
    return {
        "target_type": "generic",
        "intent": "center",
        "label_hint": "",
        "source": "vlm_tool_use_failed",
        "reasoning": f"parse_fail_{last_err}",
        "visible": True,
    }
