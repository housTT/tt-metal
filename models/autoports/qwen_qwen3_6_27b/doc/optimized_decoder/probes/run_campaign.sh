#!/usr/bin/env bash
# Serialized driver for a full re-measurement of this stage.
#
# `regenerate_evidence.sh` is the per-group script; this runs the groups in the only order that is
# correct, one at a time, never a profiler run beside a watcher run, and never two device processes at
# once - the device is exclusive and a killed process mid-operation wedges the PCIe link.
#
# It writes one stable log you can keep tailing across runs:
#
#   tail -f /home/ttuser/dev/qwen/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/logs/campaign.log
#
# Per-probe detail goes to that directory's individual logs (probe_optimized_policy.log and friends);
# this log records what started, what finished, and what failed.
#
#   usage: run_campaign.sh [group ...]      (default: probes longprobes tracy suite)
set -uo pipefail
REPO=/home/ttuser/dev/qwen/tt-metal
ART="$REPO/models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder"
LOG="$ART/logs/campaign.log"
mkdir -p "$ART/logs"
# NOT named GROUPS: that is a bash builtin array holding the caller's group IDs, and assigning to it
# silently does nothing - the loop then iterates over group IDs and every "group" is an unknown target.
STAGES=("$@")
if [ ${#STAGES[@]} -eq 0 ]; then
  STAGES=(probes longprobes tracy suite)
fi

fail=0
for group in "${STAGES[@]}"; do
  echo "== $(date -Is) START $group" >> "$LOG"
  bash "$ART/probes/regenerate_evidence.sh" "$group" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "== $(date -Is) OK $group" >> "$LOG"
  else
    echo "== $(date -Is) FAILED $group (exit $rc)" >> "$LOG"
    fail=1
  fi
done
echo "== $(date -Is) CAMPAIGN DONE (fail=$fail)" >> "$LOG"
exit "$fail"
