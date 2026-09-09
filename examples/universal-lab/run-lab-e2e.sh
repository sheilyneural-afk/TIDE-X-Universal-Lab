#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a complete Universal Lab experimental pipeline in shadow mode.

Usage:
  run-lab-e2e.sh <model-root> <receiver-request.json> <discovery-request.json> \
    <planning-request.json> <backend> <materialization-layout.json> \
  <backend-policy-or-none.json> <steering-layout-or-none.json> \
    <shadow-runner-path> <shadow-input.json> <backend-selection-input-or-policy.json> \
    <universality-input.json> <promotion-gate-request-or-policy.json> <output-dir>

backend:
  dense | low-rank | sparse | steering

The script runs fail-closed pipeline phases and stores all outputs in output-dir:
  00-lab-summary.json
  01-receiver-profile.json
  02-discovery-report.json
  03-shadow-plan.json
  04-shadow-plan-replay.json
  05-materialization-candidate.json
  06-shadow-evaluation-receipt.json
  07-backend-selection-receipt.json
  08-universality-receipt.json
  09-promotion-receipt.json

Notes:
  - For "dense" and non-steering backends, backend-policy/steering-layout slots are optional.
  - For backends that need them, provide explicit JSON files; for dense they are ignored.
  - The final gate never auto-activates; it only returns readiness evidence.
EOF
}

if [ "$#" -ne 14 ]; then
  usage
  exit 1
fi

MODEL_ROOT=$1
RECEIVER_REQUEST=$2
DISCOVERY_REQUEST=$3
PLANNING_REQUEST=$4
BACKEND=$5
MATERIALIZATION_LAYOUT=$6
BACKEND_POLICY=$7
STEERING_LAYOUT=$8
SHADOW_RUNNER=$9
SHADOW_INPUT=$10
SELECTION_INPUT=$11
UNIVERSALITY_INPUT=$12
PROMOTION_INPUT=$13
OUTPUT_DIR=$14

TIDEX_BIN=${TIDEX_BIN:-tidex}

mkdir -p "$OUTPUT_DIR"

run_and_capture() {
  local output="$1"
  shift
  local log_prefix
  log_prefix=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  echo "[$log_prefix] $*"
  "$TIDEX_BIN" "$@" > "$output"
}

case "$BACKEND" in
  dense|low-rank|sparse|steering)
    ;;
  *)
    echo "Invalid backend '$BACKEND'. Expected dense, low-rank, sparse, steering." >&2
    exit 1
    ;;
esac

run_and_capture \
  "$OUTPUT_DIR/00-lab-summary.json" \
  lab e2e \
  "$MODEL_ROOT" \
  "$RECEIVER_REQUEST" \
  "$DISCOVERY_REQUEST" \
  "$PLANNING_REQUEST" \
  "$BACKEND" \
  "$MATERIALIZATION_LAYOUT" \
  "$BACKEND_POLICY" \
  "$STEERING_LAYOUT" \
  "$SHADOW_RUNNER" \
  "$SHADOW_INPUT" \
  "$SELECTION_INPUT" \
  "$UNIVERSALITY_INPUT" \
  "$PROMOTION_INPUT" \
  "$OUTPUT_DIR"

echo "Universal Lab pipeline finished. Outputs in $OUTPUT_DIR"
