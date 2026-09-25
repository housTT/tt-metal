#!/usr/bin/env bash
# Is the chunked-prefill SDPA's long-context error worse on *peaked* attention than on flat?
#
# Stage 1 characterised the 262144-key `chunked_scaled_dot_product_attention` accuracy cliff with a
# model-free reproducer and recorded `alpha` - the device/torch output ratio - at 1.091 for 512 k
# chunks. It then wrote, of that measurement:
#
#     "Note the synthetic probe is the worst case for it - random Q/K/V give a maximally flat softmax,
#      so every chunk contributes equally.  Real attention is peakier and the layer-level effect is
#      far smaller."
#
# That sentence was never tested. This stage's real-weight full-context run says the layer-level effect
# on the real checkpoint is a 3.1 % *shrink* of the prefill tail - larger than the synthetic layer's
# 0.3 %, and the opposite sign to the isolated 1.091 - so "far smaller" is not what happens.
#
# stage 1's own probe already takes a `SCALE` knob (the score scale), and raising it is exactly how you
# make the softmax peakier without changing anything else. This sweeps it at the shipped k-chunk
# selection, so the mechanism is established model-free rather than inferred from elimination.
#
#   bash models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_sdpa_peakiness.sh
#
# Writes logs/probe_sdpa_peakiness.log.  Opens the device; run nothing else on it at the same time.
set -uo pipefail
REPO=/home/ttuser/dev/qwen/tt-metal
cd "$REPO"
source "$REPO/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/ttenv.sh" > /dev/null
ART="$REPO/models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder"
LOG="$ART/logs/probe_sdpa_peakiness.log"
PROBE="$REPO/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/probes/probe_sdpa_synthetic.py"

: > "$LOG"
{
  echo "# chunked prefill SDPA, 262144 keys, k_chunk=512 / q_chunk=64 - the shipped selection."
  echo "# SCALE is the score scale, i.e. how peaked the softmax is.  0.0625 = head_dim**-0.5, which is"
  echo "# stage 1's default and the flattest case; larger values concentrate the attention."
  echo "# alpha is the device/torch output ratio: 1.0 is exact, <1 means the device output is small."
} >> "$LOG"

for scale in 0.0625 0.125 0.25 0.5 1.0 2.0 4.0; do
  echo "== SCALE=$scale" >> "$LOG"
  LENGTHS=262144 Q_CHUNK=64 K_CHUNK=512 SCALE="$scale" python "$PROBE" 2>&1 \
    | grep -E "tail_pcc" >> "$LOG" || echo "  (no row - see the probe's own output)" >> "$LOG"
done
echo "== PEAKINESS SWEEP DONE" >> "$LOG"
