#!/usr/bin/env bash
set -uo pipefail

CLEF=/home/hous/dev/clef
AUTOPORT=$CLEF/tt-metal/models/autoports/cloudflare_clef
SCRIPTS=$AUTOPORT/scripts
EVALS=$CLEF/evals
REFERENCE=$CLEF/reports/reference
KEV=/home/hous/dev/kev/kev
HOSTRUN=$CLEF/bin/hostrun
BASE_URL=http://127.0.0.1:8008
OUT_ROOT=$CLEF/reports/eval
LOG_DIR=$CLEF/logs
LOG_PREFIX=stage5
CONC=4
LIMIT=""
KEV_RECORDS=""
STEPS="arc banking77 newyorker samples kev summarize"
KEV_SUITES="hard-v1 devtools-v1 documents-v1"
MODEL=clef
FINAL_JSON=$CLEF/reports/final_numbers.json
DOC=$AUTOPORT/doc/benchmark/EVAL.md

usage() {
  echo "usage: $0 [--base-url URL] [--out-root DIR] [--concurrency N] [--limit N] [--kev-records N]"
  echo "          [--steps \"arc banking77 newyorker samples kev summarize\"] [--kev-suites \"hard-v1 devtools-v1 documents-v1\"]"
  echo "          [--log-dir DIR] [--log-prefix stage5] [--model clef] [--final-json PATH] [--doc PATH]"
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL=$2; shift 2 ;;
    --out-root) OUT_ROOT=$2; shift 2 ;;
    --concurrency) CONC=$2; shift 2 ;;
    --limit) LIMIT=$2; shift 2 ;;
    --kev-records) KEV_RECORDS=$2; shift 2 ;;
    --steps) STEPS=$2; shift 2 ;;
    --kev-suites) KEV_SUITES=$2; shift 2 ;;
    --log-dir) LOG_DIR=$2; shift 2 ;;
    --log-prefix) LOG_PREFIX=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --final-json) FINAL_JSON=$2; shift 2 ;;
    --doc) DOC=$2; shift 2 ;;
    *) usage ;;
  esac
done

mkdir -p "$OUT_ROOT" "$LOG_DIR"
LIMIT_FLAG=()
[[ -n $LIMIT ]] && LIMIT_FLAG=(--limit "$LIMIT")
FAILURES=0

stamp() { date -u +"%Y-%m-%d %H:%M:%S UTC"; }

if ! curl -sf "$BASE_URL/v1/health" > /dev/null; then
  echo "$(stamp) server at $BASE_URL is not healthy" >&2
  exit 1
fi
echo "$(stamp) server $BASE_URL: $(curl -sf "$BASE_URL/v1/models" | python3 -c 'import json,sys; m=json.load(sys.stdin)["models"][0]; print(m["backend"], m["device"], m["precision"])')"

remote() {
  local name=$1 records=$2 output=$3
  echo "$(stamp) RUN $name: $records -> $output"
  "$HOSTRUN" python "$SCRIPTS/eval_remote.py" --base-url "$BASE_URL" --concurrency "$CONC" --model "$MODEL" \
    --records "$records" --output "$output" --summary "$OUT_ROOT/$name.summary.json" "${LIMIT_FLAG[@]}" \
    2>&1 | tee "$LOG_DIR/${LOG_PREFIX}_eval_$name.log" | grep -E "SUMMARY|FAILED|EVAL_REMOTE_DONE|server |records,"
  if grep -q "EVAL_REMOTE_DONE failed=0" "$LOG_DIR/${LOG_PREFIX}_eval_$name.log"; then
    echo "$(stamp) OK $name"
  else
    echo "$(stamp) FAILED $name (see $LOG_DIR/${LOG_PREFIX}_eval_$name.log)"
    FAILURES=$((FAILURES + 1))
  fi
}

for step in $STEPS; do
  case "$step" in
    arc) remote arc_challenge_test "$EVALS/arc_challenge_test.jsonl" "$OUT_ROOT/arc_challenge_test.jsonl" ;;
    banking77) remote banking77_test "$EVALS/banking77_test.jsonl" "$OUT_ROOT/banking77_test.jsonl" ;;
    newyorker) remote newyorker_matching_test "$EVALS/newyorker_matching_test.jsonl" "$OUT_ROOT/newyorker_matching_test.jsonl" ;;
    samples)
      for s in arc_challenge banking77 newyorker_matching; do
        remote "${s}_test_sample100" "$EVALS/${s}_test_sample100.jsonl" "$OUT_ROOT/${s}_test_sample100.jsonl"
      done
      ;;
    parity)
      remote reference_text "$REFERENCE/records_text.jsonl" "$OUT_ROOT/reference_text.jsonl"
      remote reference_image "$REFERENCE/records_image.jsonl" "$OUT_ROOT/reference_image.jsonl"
      remote dev64_text "$REFERENCE/dev64_text.jsonl" "$OUT_ROOT/dev64_text.jsonl"
      remote dev16_image "$REFERENCE/dev16_image.jsonl" "$OUT_ROOT/dev16_image.jsonl"
      ;;
    kev)
      for suite in $KEV_SUITES; do
        out=$OUT_ROOT/kev/$suite/test
        if [[ -f $out/report.json ]]; then
          echo "$(stamp) SKIP kev $suite: $out/report.json exists"
          continue
        fi
        if [[ -d $out ]]; then
          mv "$out" "$out.incomplete.$(date +%s)"
        fi
        mkdir -p "$(dirname "$out")"
        flags=(--remote "$BASE_URL" --out "$out" --remote-concurrency "$CONC" --remote-model "$MODEL")
        if [[ -n $KEV_RECORDS ]]; then
          slice=$OUT_ROOT/kev/$suite/test_head${KEV_RECORDS}.jsonl
          head -n "$KEV_RECORDS" "$KEV/evals/$suite/test.jsonl" > "$slice"
          flags+=(--data "$slice")
          echo "$(stamp) RUN kev $suite: first $KEV_RECORDS test records ($slice) -> $out"
        else
          flags+=(--suite "evals/$suite" --allow-test)
          echo "$(stamp) RUN kev $suite: test split ($(wc -l < "$KEV/evals/$suite/test.jsonl") records) -> $out"
        fi
        (cd "$KEV" && env -u VIRTUAL_ENV uv run python -m kev.benchmark "${flags[@]}") 2>&1 | tee "$LOG_DIR/${LOG_PREFIX}_kev_$suite.log" | grep -E "evaluated |\"acc\"|\"ece\"|\"brier\"|\"n\"|_records|rror|Traceback"
        if [[ -f $out/report.json ]]; then
          echo "$(stamp) OK kev $suite"
        else
          echo "$(stamp) FAILED kev $suite (see $LOG_DIR/${LOG_PREFIX}_kev_$suite.log)"
          FAILURES=$((FAILURES + 1))
        fi
      done
      ;;
    summarize)
      echo "$(stamp) RUN summarize_eval.py"
      "$HOSTRUN" python "$SCRIPTS/summarize_eval.py" --out-root "$OUT_ROOT" --final-json "$FINAL_JSON" --doc "$DOC" --write 2>&1 | tee "$LOG_DIR/${LOG_PREFIX}_summarize.log"
      ;;
    *) echo "unknown step $step" >&2; usage ;;
  esac
done

echo "$(stamp) RUN_EVAL_DONE failures=$FAILURES out_root=$OUT_ROOT"
exit $(( FAILURES > 0 ? 1 : 0 ))
