# P150x4 penalty slot-remap probe

- Timestamp: 2026-08-31 19:14 UTC
- Workspace: `/home/ttuser/dev/gpt-oss-20b`
- Shape/dtype/layout: `[32, 262144]`, INT32, TILE, DRAM
- Remap: `[3, 0, 1, 2, 4, ..., 31]`
- Operation: persistent UINT16 expanded index, `ttnn.gather(..., dim=0)`, then `ttnn.copy` into the original buffer inside a transient-allocation scope.

Results:

| Placement | Reconstructed shape | Buffer addresses unchanged | Exact equality | Max absolute error |
|---|---:|---:|---:|---:|
| vocab-sharded `(None, 1)` | `[32, 262144]` | yes | yes | 0 |
| replicated `(None, None)` | `[32, 262144]` | yes | yes | 0 |

All four module origins were verified before opening the mesh. The probe closed the P150x4 mesh cleanly.
