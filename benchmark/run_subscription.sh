#!/bin/bash
# Run benchmark in subscription-only mode.
#
# Background: claude CLI auth precedence picks ANTHROPIC_API_KEY over the
# Max/Pro OAuth subscription when both are present. Even with grid_core.py's
# surgical env strip in _vlm_call_cli, running the Python interpreter with
# the API key visible risks leaking back into Anthropic SDK calls or any
# future code that imports anthropic directly. This wrapper strips the auth
# env vars at the shell level so the Python process and all its children
# see a clean, subscription-only environment.
#
# Usage:
#   ./benchmark/run_subscription.sh --methods grid,pinpoint --use-cli
#
# Any args are forwarded to run_benchmark.py.

set -e

unset ANTHROPIC_API_KEY
unset ANTHROPIC_AUTH_TOKEN

echo "=== Subscription-only benchmark mode ==="
echo "Auth status after unset:"
claude auth status 2>&1 | grep -E '"(loggedIn|authMethod|subscriptionType|apiKeySource|orgName)"' || true
echo ""
echo "Starting: python -u benchmark/run_benchmark.py $*"
echo ""

exec python -u benchmark/run_benchmark.py "$@"
