# Rate-limit Smoke Check Protocol (decision 044)

Pre-flight checklist for `benchmark/run_benchmark.py --use-cli` runs, enforced
from v7 onwards. This protocol is **separate from and complementary to** the
decision 032 "90-second smoke check" (which detects cascading silent failure
in the first benchmark case); this one catches **auth/credit/rate-limit
misconfiguration before a long run starts**.

Reference: `dlg/benchmark/msg-2026-04-11T15:49:03.046-1745` (decision 044),
`dlg/benchmark/msg-2026-04-11T15:44:47.985-403a` (pitfall root cause).

## Why this exists

The v6 benchmark run (2026-04-11 13:41 → 18:52 JST, 5h 11m, 100 runs) was
billed entirely against Anthropic API credits despite the user holding a
Claude Max subscription. Root cause: `claude` CLI auth precedence picks
`ANTHROPIC_API_KEY` (env) over OAuth subscription when both are present.

commit `a131122` installs a two-layer fix:

- **Layer 1** (`benchmark/grid_core.py::_vlm_call_cli`): strip
  `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` from the `claude -p` subprocess
  env, forcing OAuth fallback. Parent Python env untouched.
- **Layer 2** (`benchmark/run_subscription.sh`): shell wrapper that unsets
  the same vars at parent shell level. Belt-and-braces when the Python
  process itself must be subscription-only.

This protocol verifies both layers work and the subscription is not about to
hit a rate limit, **before** committing hours of wall-clock time.

## Protocol

Run these steps in order. **Abort and escalate to supervisor if any step
fails.**

### Step 1 — Auth state verification (<10 s)

```bash
claude auth status
```

**Required output fields**:

| field | expected value | what it means |
|---|---|---|
| `loggedIn` | `true` | user is OAuth-logged into claude.ai |
| `authMethod` | `claude.ai` or `oauth_token` | OAuth or long-lived CLI token |
| `subscriptionType` | `"max"` or `"pro"` | active subscription plan |
| `apiKeySource` | **absent** or `"CLAUDE_CODE_OAUTH_TOKEN"` | **must NOT be** `"ANTHROPIC_API_KEY"` |
| `orgName` | non-null | account is resolved |

**Gotcha**: if `apiKeySource: "ANTHROPIC_API_KEY"` appears, the current shell
still has the env var set and auth precedence will pick API key over
subscription. Either:
- `unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN` in the current shell, or
- switch to a fresh shell where the vars are not set, or
- use `benchmark/run_subscription.sh` which handles this automatically.

**Gotcha**: if `subscriptionType: null` appears despite `loggedIn: true`, you
are likely still in API-key mode (the CLI hides subscription info when using
an API key). Re-verify after unsetting env vars.

### Step 2 — Env state pre-check (<5 s)

```bash
if [ -n "$ANTHROPIC_API_KEY" ]; then
  echo "WARNING: ANTHROPIC_API_KEY is set in current env (length=${#ANTHROPIC_API_KEY})"
  echo "Layer 1 (grid_core.py env strip) will handle this for claude -p subprocess."
  echo "But the Python process itself still sees the key."
fi
if [ -n "$CLAUDE_CODE_OAUTH_TOKEN" ]; then
  echo "OK: CLAUDE_CODE_OAUTH_TOKEN is set (length=${#CLAUDE_CODE_OAUTH_TOKEN})"
fi
```

**Decision tree**:

- `ANTHROPIC_API_KEY` set + `CLAUDE_CODE_OAUTH_TOKEN` set
  → prefer `benchmark/run_subscription.sh` wrapper (Layer 2) for cleanest env
- `ANTHROPIC_API_KEY` set + `CLAUDE_CODE_OAUTH_TOKEN` not set
  → Layer 1 alone is sufficient (`grid_core.py` strips the key in subprocess);
    subscription OAuth will be used via auth precedence fallback #6
- Neither set
  → default OAuth subscription; Layer 1 is still active but no-op

### Step 3 — Single `claude -p` smoke call (<60 s)

Minimum viable `claude -p` invocation with the same flags the benchmark uses:

```bash
echo "test" | claude -p "Reply with exactly one word: OK" \
  --setting-sources user \
  --no-session-persistence \
  --model sonnet \
  2>/tmp/smoke_stderr.txt
```

**Expected**: stdout contains `"OK"`, exit code 0.

**Stderr should be empty or contain only expected hook warnings**. Stderr
pollution with auth errors like `Credit balance is too low` or
`Authentication failed` is a hard abort signal — do NOT proceed to Step 4.

Also consider running the modified `_vlm_call_cli` function directly via a
Python one-liner to verify the env strip code path works end-to-end:

```bash
PYTHONIOENCODING=utf-8 python -c "
import sys, base64, io, os
sys.path.insert(0, 'benchmark')
from PIL import Image
import grid_core
img = Image.new('RGB', (100, 100), (255, 0, 0))
buf = io.BytesIO(); img.save(buf, 'JPEG', quality=80)
b64 = base64.b64encode(buf.getvalue()).decode('ascii')
ans = grid_core._vlm_call_cli(
    b64_image=b64,
    system='You identify the dominant color of an image.',
    user='What is the dominant color? Reply with one word: red, blue, or green.',
    model='sonnet',
)
print('parent env key still set:', bool(os.environ.get('ANTHROPIC_API_KEY')))
print('VLM answer:', repr(ans))
assert ans and 'red' in ans.lower(), 'smoke fail'
print('SMOKE PASS')
"
```

**Expected**: prints `SMOKE PASS`, parent env unchanged, VLM correctly
identifies the red test image.

### Step 4 — 10-case smoke run (30–60 min)

Run a small subset of the benchmark to probe rate-limit behavior:

```bash
python -u benchmark/run_benchmark.py \
  --methods grid,pinpoint \
  --use-cli \
  --categories ui_button \
  2>&1 | tee benchmark/smoke_v7.log
```

This triggers ~20 `claude -p` calls per case × 10 cases = **~200 CLI calls**
in 30–60 min wall clock. Enough to hit a short-term rate limit if one exists
at the subscription tier, but small enough that a failure does not cost
hours.

Alternative (Layer 2 wrapper):

```bash
bash benchmark/run_subscription.sh --methods grid,pinpoint --use-cli --categories ui_button
```

### Step 5 — Post-smoke diagnostics (<5 min)

Check the smoke run output:

```bash
echo "=== partial row count ==="
wc -l benchmark/results.json.partial

echo "=== WARNING / rc=1 / Credit count ==="
grep -c "WARNING\|rc=1\|Credit\|Authentication" benchmark/smoke_v7.log

echo "=== last 20 lines ==="
tail -20 benchmark/smoke_v7.log
```

**Pass criteria**:

- `results.json.partial` has **20 rows** (10 cases × 2 methods × 1 repeat)
- WARNING / rc=1 / Credit / Authentication count = **0**
- No `None, None` FAIL rows (all predictions are non-null coordinates)
- Smoke log ends with the partial `BENCHMARK SUMMARY` block

**Fail signatures**:

| symptom | likely cause | action |
|---|---|---|
| Early exit with `Credit balance is too low` | env strip not effective, hitting API | abort, verify Step 1 / Step 2 |
| All predictions `None, None`, `Q=0`, `t=0s` | subprocess rc=1 at every call | abort, check stderr in smoke log |
| Partial file grows but rate slows dramatically | subscription rate-limit soft-throttle | pause, wait 15 min, resume with `--resume` |
| Intermittent FAILs but most pass | model refusal or prompt edge case | continue, investigate offline |

### Step 6 — Full run decision

If Step 5 passes cleanly, launch the full run:

```bash
# unchanged from the working command:
python -u benchmark/run_benchmark.py --methods grid,pinpoint --use-cli \
  2>&1 | tee benchmark/full_run_v7.log
```

The existing decision 032 90-second smoke check still applies: monitor the
partial file at 90–120 s after start to verify the first case produced a
non-null row.

If rate-limit symptoms appear mid-run, interrupt and resume with `--resume`
after a cooldown. The JSONL partial writes (`811028c`) are per-case fsync'd,
so partial progress is never lost.

## When to skip this protocol

Only skip Step 4 (the 10-case smoke run) when:

1. A v6-equivalent run has successfully completed in the **same session /
   shell env** within the past 2 hours, AND
2. No code changes have been made to `grid_core.py` or `run_benchmark.py`
   in that window, AND
3. `claude auth status` currently reports the same auth state as at the
   start of that successful run

In all other cases, run the full 6-step protocol. The cost (60 min of
smoke-run subscription usage) is small compared to the cost of discovering a
misconfiguration 3 hours into a 5-hour run.

## History

- **2026-04-11** — v6 run (13:41 → 18:52 JST, 5h 11m) executed without this
  protocol; discovered post-hoc that 100% of API-key traffic was billed
  against Anthropic credits while user held an unused Claude Max
  subscription. commit `a131122` added Layer 1 + Layer 2 env stripping.
  decision 044 mandated this protocol for all future runs.
- **2026-04-11** — User set `CLAUDE_CODE_OAUTH_TOKEN` at Machine scope,
  resolving Q2 of decision 044 (long-lived CLI OAuth token in env).
- **v7** — First run scheduled to use this protocol. Expected after Layer 1
  OCR-first implementation (Branch A next phase).
