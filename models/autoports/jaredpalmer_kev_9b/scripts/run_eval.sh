#!/usr/bin/env bash
set -euo pipefail

KEV=/home/hous/dev/kev/kev
OUT_ROOT=/home/hous/dev/kev/reports/eval
BASE_URL=""
CONC=4
TEST=0
MODEL=kev-latest
SUITES="hard-v1 devtools-v1 documents-v1 breadth-v1 smoke-v1"

usage() {
  echo "usage: $0 --base-url URL [--concurrency N] [--test] [--suites \"a b\"] [--out-root DIR] [--model NAME]"
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL=$2; shift 2 ;;
    --concurrency) CONC=$2; shift 2 ;;
    --test) TEST=1; shift ;;
    --suites) SUITES=$2; shift 2 ;;
    --out-root) OUT_ROOT=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n $BASE_URL ]] || usage

cd "$KEV"

available() {
  uv run python - "$1" "$2" "$3" <<'EOF'
import sys
from kev.suite import load_split
suite, split, allow = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
try:
    print(f"AVAILABLE {len(load_split(suite, split, allow_test=allow))}")
except Exception as e:
    print(f"NOT-AVAILABLE {type(e).__name__}: {str(e)[:300]}")
EOF
}

for suite in $SUITES; do
  splits="development"
  [[ $TEST == 1 ]] && splits="development test"
  for split in $splits; do
    out=$OUT_ROOT/$suite/$split
    allow=0
    [[ $split == test ]] && allow=1
    status=$(available "evals/$suite" "$split" "$allow" 2>&1 | tail -1)
    if [[ $status != AVAILABLE* ]]; then
      echo "NOT-AVAILABLE $suite/$split: ${status#NOT-AVAILABLE }"
      continue
    fi
    if [[ -f $out/report.json ]]; then
      echo "SKIP $suite/$split: $out/report.json exists"
      continue
    fi
    if [[ -d $out ]]; then
      mv "$out" "$out.incomplete.$(date +%s)"
    fi
    mkdir -p "$(dirname "$out")"
    flags=(--remote "$BASE_URL" --suite "evals/$suite" --out "$out" --remote-concurrency "$CONC" --remote-model "$MODEL")
    [[ $split == test ]] && flags+=(--allow-test)
    echo "RUN $suite/$split (${status#AVAILABLE } records) -> $out"
    uv run python -m kev.benchmark "${flags[@]}" 2>&1 | tee "$out.log"
  done
done
