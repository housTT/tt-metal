# AutoDebug: batch-two precision policy boundary

> **Superseded diagnosis (2026-08-28):** This report is retained as the
> fresh-context diagnosis that established `None` as the unambiguous automatic
> sentinel, but its BF16-default recommendation is historical. Final stage
> review first promoted BFP8 over BF16, then the cumulative final-32-core matrix
> in `AUTOFIX_attention_precision_final32c.md` promoted BFP4/LoFi for configured
> batch 2 through 32. Omission now resolves BFP8/HiFi2 at configured batch 1 and
> BFP4/LoFi at multibatch; explicit BFP8/BF16 requests remain exact.

## Headline finding

The batch-two discrepancy is an API/test-harness ambiguity, not evidence that
the production batch-two policy should revert to BFP8 or that the PCC bar
should change.

`OptimizedDecoder.from_state_dict()` currently defaults `policy` to the
`ATTENTION_BFP8_POLICY` object and then treats identity with that same object as
the signal for an *omitted/automatic* request:

```python
policy=DEFAULT_OPTIMIZED_POLICY
...
requested_policy = policy
if policy is DEFAULT_OPTIMIZED_POLICY and max_batch_size > 1:
    policy = ATTENTION_BF16_CONTROL
```

Consequently, an explicit `policy=ATTENTION_BFP8_POLICY` is indistinguishable
from omitting `policy`; both are silently rewritten to BF16 when
`max_batch_size > 1`. This affects both the public constructor boundary and the
tests' ability to run an explicit BFP8 structural control.

The evidence supports keeping automatic multi-batch production selection at
BF16:

| Case | Effective attention policy | Decode PCC / result |
| --- | --- | --- |
| Synthetic batch two, older run | BFP8 | sliding `0.991879`, full `0.995540`; pass |
| Synthetic batch two, current run | automatic BF16 | sliding `0.987673715`; fail; full pass |
| Real checkpoint, FullLocal batch two | BFP8 | `0.975027444`; fail |
| Real checkpoint, FullLocal batch two | automatic BF16 | `0.991735289`; pass |

The unchanged `0.99` decode threshold is doing useful work. The real-weight
result rejects BFP8 as the automatic production policy for batch two, while
the synthetic BFP8/BF16 reversal is a diagnostic distribution discrepancy.
The `$optimize` precision rule explicitly says synthetic/random PCC must not
veto a real-weight policy win; synthetic data remains valuable for structural,
paging, trace, determinism, and runtime checks.

## Why the current tests obscure the boundary

- `_run_synthetic_gate()` does not pass a policy. After automatic multi-batch
  BF16 selection was added, its batch-two sliding case changed precision and
  ceased to be the same structural control that had passed under BFP8.
- `_run_real_checkpoint_gate()` defaults its own `policy` argument to
  `DEFAULT_OPTIMIZED_POLICY` and always writes that object into constructor
  kwargs. Its assertion then expects BF16 when the requested/default object is
  used at larger batch. This passes, but cannot prove that constructor omission
  and explicit BFP8 are distinct API choices.
- `decoder.requested_policy` currently records the BFP8 object even when the
  caller intended no explicit policy and the effective policy is BF16. Thus it
  does not accurately record caller intent.

The layer-kind split is consistent with a marginal synthetic precision effect:
the BF16 synthetic full-attention node passes while synthetic sliding attention
misses by about `0.00233`. The supplied evidence does not point to a changed
paged-cache, trace, or FullLocal semantic contract. Those contracts still need
to be rerun after the boundary fix, but they should not be repaired by lowering
PCC.

## Prioritized fix

1. Use `None` as the sole automatic-policy sentinel.

   ```python
   policy: OptimizedDecoderPolicy | None = None
   requested_policy = policy
   if policy is None:
       policy = ATTENTION_BF16_CONTROL if max_batch_size > 1 else DEFAULT_OPTIMIZED_POLICY
   elif policy not in SUPPORTED_POLICIES:
       raise ValueError(...)
   ```

   This makes the contract unambiguous:

   - omitted/`None`, batch one -> canonical BFP8/HiFi2;
   - omitted/`None`, batch greater than one -> BF16/HiFi2;
   - explicit BFP8 at any batch -> BFP8/HiFi2, with no implicit rewrite;
   - other explicit candidate policies -> exactly the requested candidate.

   Keep `decoder.policy` as the effective policy and
   `decoder.requested_policy` as the caller's request (`None` for automatic).
   Cache naming is already derived from the effective policy dtype, so this
   change should preserve cache isolation.

2. Separate production-default evidence from the synthetic structural control.

   - Change `_run_real_checkpoint_gate()` to default its helper `policy` to
     `None` and only put `policy` in kwargs when it is explicitly provided.
     The real FullLocal batch-two gate must omit the policy, assert
     `requested_policy is None`, assert effective BF16, and retain the `0.99`
     PCC bar. This is the production policy gate.
   - Let `_run_synthetic_gate()` accept an optional explicit policy. Invoke the
     batch-two synthetic sliding/full test with
     `policy=ATTENTION_BFP8_POLICY`, and assert both requested and effective
     policy are BFP8. Label it a synthetic structural control; do not present it
     as proof of the production batch-two dtype.
   - Keep batch-32 production/default coverage on omitted policy (effective
     BF16), and keep batch-one default coverage on omitted policy (effective
     BFP8).
   - Update static attestations so they prove the constructor default is
     `None`, explicit BFP8 is not rewritten, and automatic resolution is
     batch-aware.

3. Document the two contracts explicitly in the work log: production
   automatic policy is selected from batch capacity; explicit policy means
   exactly that candidate. Record the synthetic BF16 miss as a reconciled
   diagnostic, not a passing gate and not a reason to weaken acceptance.

## Required verification after implementation

Run these with the existing unchanged PCC thresholds:

1. Host/static policy-boundary tests:
   omitted batch one -> BFP8; omitted batch two/32 -> BF16; explicit BFP8 at
   batch two -> BFP8; unsupported explicit policy -> `ValueError`.
2. Real checkpoint FullLocal batch-two paged prefill plus traced decode with no
   policy kwarg. Require effective BF16, `requested_policy is None`, decode PCC
   at least `0.99`, deterministic replay, and no fallback.
3. Synthetic batch-two sliding and full structural controls with explicit BFP8.
   Require requested/effective BFP8, the same `0.99` PCC threshold, paged-cache
   behavior, and deterministic traced replay.
4. Real batch-one sliding/full default gates with no policy kwarg. Require
   effective BFP8 and unchanged PCC/performance.
5. Batch-32 default capacity/replay gate with no policy kwarg. Require effective
   BF16 and the already established bitwise replay behavior.
6. Candidate/performance tests must continue passing policy explicitly so
   measurements cannot be silently changed by batch-aware automatic selection.

## Investigation limitation

The mandated fresh AutoDebug runner was launched in an isolated docs
subdirectory so it could not overwrite the unrelated repo-root
`AUTODEBUG.md`. Its nested Codex sandbox could not initialize loopback/user
namespace support and therefore could not read the checkout. The findings
above come from a direct source/log audit after that runner was stopped; no
hardware commands and no implementation or test edits were made.
