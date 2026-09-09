#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Validate outputs produced by run-lab-e2e.sh.

Usage:
  check-lab-e2e.sh <output-directory> [--strict]

The script checks that expected files exist and, when jq is installed,
validates their schema fields. With --strict it exits non-zero for any
schema mismatch, including JSON parse failures.
EOF
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  usage
  exit 1
fi

OUTPUT_DIR=$1
STRICT=${2:-}
STRICT_SCHEMA=false
if [ "${2:-}" = "--strict" ]; then
  STRICT_SCHEMA=true
fi

has_jq=true
if ! command -v jq >/dev/null 2>&1; then
  has_jq=false
  echo "warning: jq not installed; only file existence checks will run"
fi

fail() {
  local message=$1
  echo "FAIL: $message"
  exit 1
}

check_file() {
  local path=$1
  if [ ! -f "$path" ]; then
    fail "missing file: $path"
  fi
  if [ ! -s "$path" ]; then
    fail "empty file: $path"
  fi
  if [ "$has_jq" != "true" ]; then
    return 0
  fi
  if ! jq -e '.schema | type == "string"' "$path" >/dev/null; then
    if [ "$STRICT_SCHEMA" = true ]; then
      fail "invalid json or schema field: $path"
    else
      echo "warning: cannot read schema field in $path"
      return 0
    fi
  fi
}

assert_schema() {
  local path=$1
  local expected=$2
  if [ "$has_jq" != "true" ]; then
    return 0
  fi
  if ! jq -e --arg expected "$expected" '.schema == $expected' "$path" >/dev/null; then
    if [ "$STRICT_SCHEMA" = true ]; then
      fail "schema mismatch in $path (expected $expected)"
    else
      echo "warning: schema mismatch in $path"
    fi
  fi
}

if [ ! -d "$OUTPUT_DIR" ]; then
  fail "output directory not found: $OUTPUT_DIR"
fi

declare -a files=(
  "$OUTPUT_DIR/00-lab-summary.json:cerebro.tidex.lab_e2e_summary/v1"
  "$OUTPUT_DIR/01-receiver-profile.json:cerebro.tidex.inspected_receiver_artifacts/v1"
  "$OUTPUT_DIR/02-discovery-report.json:cerebro.tidex.capability_discovery_report/v1"
  "$OUTPUT_DIR/03-shadow-plan.json:cerebro.tidex.universal_capability_shadow_plan_receipt/v1"
  "$OUTPUT_DIR/04-shadow-plan-replay.json:cerebro.tidex.universal_shadow_plan_replay/v1"
  "$OUTPUT_DIR/05-materialization-candidate.json"
  "$OUTPUT_DIR/06-shadow-evaluation-receipt.json:cerebro.tidex.shadow_evaluation_receipt/v1"
  "$OUTPUT_DIR/07-backend-selection-receipt.json:cerebro.tidex.backend_selection_receipt/v1"
  "$OUTPUT_DIR/08-universality-receipt.json:cerebro.tidex.universality_evidence_receipt/v1"
  "$OUTPUT_DIR/09-promotion-receipt.json:cerebro.tidex.universal_promotion_gate_receipt/v1"
)

for entry in "${files[@]}"; do
  path="${entry%%:*}"
  schema="${entry##*:}"
  check_file "$path"
  if [ "$schema" != "$path" ]; then
    # shellcheck disable=SC2001
    assert_schema "$path" "$schema"
  fi
done

if [ "$has_jq" = true ]; then
  echo "=== summary ==="
  candidate_schema=$(
    jq -r '.schema // "<missing>"' "$OUTPUT_DIR/05-materialization-candidate.json"
  )
  echo "- materialization plan strategy: $(jq -r 'if .shadow_plan.materialization_plan.strategy==null then "<missing>" else .shadow_plan.materialization_plan.strategy end' \
    "$OUTPUT_DIR/03-shadow-plan.json")"
  echo "- materialization-candidate schema: $candidate_schema"
  echo "- selection strategy: $(jq -r 'if .selected_strategy==null then "<missing>" else .selected_strategy end' \
    "$OUTPUT_DIR/07-backend-selection-receipt.json")"
  echo "- universality N: $(jq -r 'if .universality_n==null then "<missing>" else (.universality_n|tostring) end' \
    "$OUTPUT_DIR/08-universality-receipt.json")"
  echo "- gate readiness: $(jq -r 'if .readiness==null then "<missing>" else .readiness end' \
    "$OUTPUT_DIR/09-promotion-receipt.json")"
  echo "- authorizes_activation: $(jq -r 'if .authorizes_activation==null then "<missing>" else (.authorizes_activation|tostring) end' \
    "$OUTPUT_DIR/09-promotion-receipt.json")"
fi

if [ "$has_jq" = true ] && [ -f "$OUTPUT_DIR/09-promotion-receipt.json" ]; then
  blockers_count=$(jq -r '.blockers | length // 0' "$OUTPUT_DIR/09-promotion-receipt.json")
  echo "- gate blockers: ${blockers_count}"
  if [ "$blockers_count" -gt 0 ]; then
    echo "  blockers:"
    jq -r '.blockers[]?' "$OUTPUT_DIR/09-promotion-receipt.json"
  fi
fi

echo "Output validation complete: $OUTPUT_DIR"
