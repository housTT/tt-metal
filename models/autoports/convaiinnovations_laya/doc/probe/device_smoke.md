# Device smoke (stage 0)

Host qb2-120-p11t01, 2026 Oct 5. Kernel 7.0.0-34-generic (rebooted 2026 Oct 3), TT-KMD 2.10.0, firmware bundle 19.15.0, tt-smi 6.1.0.

First attempt 20:51 to 20:54 UTC: every chip (0 to 3) and the 1x4 mesh failed to open with `Device 0: Timed out while waiting for active ethernet core 29-25 to become active again. Try resetting the board` (TT_THROW at tt_metal/llrt/llrt.cpp:594), log `/home/hous/dev/laya/logs/p0_mesh_smoke_diag.log`.

After `tt-smi -r` (`/home/hous/dev/laya/bin/reset-devices.sh`, 20:58 UTC) every open succeeded, log `/home/hous/dev/laya/logs/p0_reset_smoke_20261005T205841Z.log`:

```
```

Each single-chip open used `TT_METAL_VISIBLE_DEVICES=<chip>` and `ttnn.open_mesh_device(MeshShape(1,1), trace_region_size=0)`; the 1x4 open used `MeshShape(1,4)` with no fabric config. Script: `/home/hous/dev/laya/bin/mesh-smoke.py`.
