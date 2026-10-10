# Full-attention profiler source-hash collision

Read-only source and artifact diagnosis, 2026-10-10. The inspecting agent ran no device jobs and changed no production source. Both failed captures remain intact.

The first warmed-prefill profiler drain failed in `tt_metal/impl/profiler/profiler.cpp:376`. Two **active** SDPA markers have the same 16-bit hash, 2450, under the original checkout prefix `/home/hous/dev/qwen3.5-35b-a3b/tt-metal`:

| Marker | Source under that prefix | Original hash | Corrected alias hash |
| --- | --- | ---: | ---: |
| `Reduce max` | `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp:1691` | 2450 | 27031 |
| `Softmax(Q@KT)@V` | Same file, line 1770 | 2450 | 34925 |

Both strings occur at lines 473–474 of each failed capture's **new** `new_zone_src_locations.log`; this is not stale/global metadata pollution. There are 31 distinct emitted zone strings, with this one collision. `kernel_profiler.hpp:38–43` constructs each identity from name, `__FILE__`, line number and `KERNEL_PROFILER`. `profiler.cpp:346–352` applies FNV-1a and XOR-folds to 16 bits; lines 374–376 reject distinct strings sharing an ID. Clearing an old zone map cannot resolve this active pair. Cached JIT objects also re-emit their source metadata (`tt_metal/jit_build/build.cpp:1013–1035,1160–1162`).

| Attempt | Preserved evidence | Result |
| --- | --- | --- |
| Original source prefix | `full_profile_hash_collision.log`; `tracy/full_attention_hash_collision/.logs/`; `profile_commands_first_retry.json` | Exit 4 at profiler drain; driver shutdown completed. |
| Runtime-root alias, original checkout CWD | `full_profile_alias_cwd_failure.log`; `tracy/full_attention_alias_cwd_failure/.logs/`; `profile_commands_second_retry.json` | Exit 4 with the same collision. The diagnostic header used the alias, but the two actual marker strings retained the original source prefix. Driver shutdown completed. |

The second failure follows the source resolver: `tt_metal/impl/kernels/kernel.cpp:53–73` searches process CWD before `TT_METAL_KERNEL_PATH`, and `tt_metal/jit_build/genfiles.cpp:69` embeds the resolved absolute source path in a generated `#include`. Setting only the runtime-root alias while running from checkout root therefore leaves these kernel identities unchanged.

The corrected invocation uses a symlink `/tmp/q35-functional-profiler-root` targeting the exact checkout, both `TT_METAL_RUNTIME_ROOT` and `TT_METAL_KERNEL_PATH` set to that alias, and fresh `TT_METAL_CACHE=/tmp/q35-functional-profiler-cache-cwd`. Its CWD is this evidence directory; checkpoint, interpreter and output paths are absolute. `tests/run_profiles.py` records the exact command, CWD and environment in `profile_commands.json` when the run finishes. `rtoptions.cpp:352–354` preserves the runtime-root spelling; lines 498–509 use lexical normalization, not symlink canonicalization, for kernel/cache paths. The effective cache is the supplied cache directory plus `/tt-metal-cache`.

Actual corrected capture metadata in `tracy/full_attention/.logs/new_zone_src_locations.log:473–474` now uses the alias for both markers, producing 27031 and 34925. Offline recomputation over all 31 emitted identities finds **zero collisions**. This verifies source identity repair, not completed performance measurement; the parent agent's active profile run and postprocessing remain the completion checks. The workaround changes profiler identities without changing kernel math or removing markers. Full layer measurement windows, between-pass drains and the 32768-op profiler buffer setting remain in place.

Closure: the corrected full capture and postprocessing both subsequently exited 0.
`profile_commands.json`, `profile_integrity.json`, and the full-attention
`measurement.json` bind complete timestamp coverage to the accepted measurement.
