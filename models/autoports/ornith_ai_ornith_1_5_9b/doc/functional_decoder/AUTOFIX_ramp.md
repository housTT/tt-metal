# AutoFix: persistent position-ramp ownership

## Starting evidence

Source-only diagnosis: `AUTODEBUG.md`, finding H1. Original failure:
`logs/trace_continuation.log`, second prefill call in linear-attention
`test_prefill_continuation[...63...]`: `pos_ramp` has no allocated buffer.
The original broad selector was `continuation or contract_extensions or decode_pcc`.
The hypothesis is that a full-extent `ttnn.slice` aliases persistent storage and
unconditional default-force deallocation frees that storage.

## Hypothesis experiment

Hardware access was granted after the coordinator's native-prefill run closed its
mesh; these two commands ran serially on one Blackhole chip. No reset was needed.
The exact pre-edit probe command was:

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 timeout 120 python -u - <<'PY' > models/autoports/ornith_ai_ornith_1_5_9b/doc/functional_decoder/logs/ramp_alias_probe.log 2>&1
import json
import torch
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
try:
    for size in (128, 256):
        source = ttnn.from_torch(torch.arange(size, dtype=torch.float32).reshape(1, size, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        sliced = ttnn.slice(source, [0, 0, 0], [1, 128, 1])
        keep = ttnn.typecast(ttnn.lt(sliced, 63.0), ttnn.float32)
        ttnn.synchronize_device(mesh)
        result = {'source_shape': list(source.shape), 'slice_shape': list(sliced.shape), 'source_address': source.buffer_address(), 'slice_address': sliced.buffer_address(), 'allocated_before': source.is_allocated()}
        ttnn.deallocate(sliced)
        result['allocated_after'] = source.is_allocated()
        print('RAMP_ALIAS_PROBE ' + json.dumps(result), flush=True)
        if size == 128:
            assert result['source_address'] == result['slice_address'] and not result['allocated_after']
        else:
            assert result['source_address'] != result['slice_address'] and result['allocated_after']
            ttnn.deallocate(source)
        ttnn.deallocate(keep)
    print('H1_VERIFIED', flush=True)
finally:
    ttnn.close_mesh_device(mesh)
PY
```

Result: exit 0, `H1_VERIFIED`. For the exact `[1,128,1]` ramp, source and
slice addresses were both `5849728`; source allocation changed `true → false`.
For the `[1,256,1]` control sliced to `[1,128,1]`, addresses were `5850752`
and `5854848`; source allocation remained `true`. The mask calculation matches
`_gdn_gates`; this is a tensor-lifetime probe, not a decoder precision experiment.

**Verdict: H1 verified.** After verification, the only implementation change was:

```diff
-            ramp = ttnn.slice(self.w["pos_ramp"], [0, 0, 0], [1, t, 1])
+            ramp, ramp_owned = _slice_owned(self.w["pos_ramp"], [0, 0, 0], [1, t, 1])
             keep = ttnn.typecast(ttnn.lt(ramp, float(logical_len)), ttnn.float32)
-            ttnn.deallocate(ramp)
+            if ramp_owned:
+                ttnn.deallocate(ramp)
```

## Verification

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 ORNITH_WEIGHTS=real pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_functional_decoder.py -k 'prefill_continuation and linear_attention' -x -v -s > models/autoports/ornith_ai_ornith_1_5_9b/doc/functional_decoder/logs/ramp_continuation_fix.log 2>&1
```

Exit 0: **4 passed, 72 deselected**, 7.22 seconds. Actual target config and real
layer-0 checkpoint weights; two prefill calls cover all 384 tokens. HF-vs-TTNN PCC:

| Split | PCC |
| --- | --- |
| 63 (original failure) | 0.999329 |
| 65 | 0.999348 |
| 128 (aligned control) | 0.999342 |
| 129 | 0.999328 |

All exceed the unchanged 0.995 bar. The formerly failing masked second calls
complete, proving persistent-ramp reuse through the decoder path. No test
implementation was changed by this hypothesis agent.

## Final status

H1 fixed with focused runtime evidence. No precision, recurrence, chunking, cache,
or threshold changes. Hardware lane returned after exit 0 and device closure.
The coordinator owns the remaining full-attention continuation/broad regression,
watcher, profiling, and final stage review. These were not run by this agent; this
report does not imply stage acceptance. No commits were created by this agent.
