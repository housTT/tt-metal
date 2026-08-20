# Split-conv remediation safety ledger

Historical limitation: the exact terminal transcript for the original split-conv hang recovery is unavailable. The retained triage proves the device-3 zero-heartbeat failure, but the original candidate PID, termination command, bounded list/reset/list outputs, lock decision, second-reset decision, and mesh-smoke output were not retained. Exact historical PIDs must therefore not be inferred.

This directory records a fresh bounded recovery and health sequence captured on 2026-08-19 before the like-for-like 16-KiB L1_SMALL remediation runs. A live unrelated Laguna vLLM process (`PID 512455`) was observed and was neither killed nor reset. The free P300 board with board ID `000004613192404c` comprises BDFs `0000:01:00.0` and `0000:02:00.0`; both chip-in-use locks had no owner. The unrelated board (`0000:03:00.0`, `0000:04:00.0`) was excluded from reset and all subsequent runs.

No stale Qwen pytest, Tracy collection, device-profiler job, EngineCore, or vLLM process from this remediation existed. No lock was deleted or cleared. The persistent Tracy web viewer does not own TT hardware and was left running.

The initial single-BDF smoke was a controlled discovery failure: filtering one half of a dual-chip P300 produces a custom one-chip topology. The installed runtime closed without opening a mesh. The board-pair retry is the authoritative smoke and exited zero.

Second reset decision: no second recovery reset was needed after the board-pair reset because the reset exited zero, all four host chips remained visible, and the installed-runtime 1x1 mesh opened and closed successfully on the isolated board pair.

Artifact inventory:

- `process_and_lock_preflight.txt`: relevant process and chip-lock ownership inspection.
- `list_before.txt`: bounded pre-reset list command/output/status.
- `reset_target_0.txt`: first targeted single-BDF reset command/output/status.
- `list_after_target_0.txt`: bounded list after that reset.
- `single_bdf_smoke_failure.txt`: exact failure classification and key output for the rejected one-chip visibility route.
- `board_pair_preflight.txt`: second owner check before resetting the topology-complete free board.
- `reset_board_pair.txt`: targeted two-BDF board reset command/output/status.
- `list_after_board_pair.txt`: bounded post-reset list command/output/status.
- `mesh_smoke_board_pair.txt`: successful installed-runtime isolated-board 1x1 mesh open/close command/output/status.
- `mesh_smoke_board_pair_full.log`: full stdout/stderr plus exit status from the final successful installed-runtime smoke rerun.
- `postflight.txt`: bounded final list, hardware-process scan, and selected-board lock ownership.
- `postflight_list_full.log`: full stdout/stderr plus exit status from the bounded final list.

All later correctness and profiler commands use `TT_VISIBLE_DEVICES=0000:01:00.0,0000:02:00.0`; the pytest `mesh_device=1` fixture opens one logical chip from that verified board-pair visibility set.
