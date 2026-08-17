tt-triage for the hang described in work_log.md §5.2 (logs/probe_bisect.py --order after wedging the
mesh instead of returning a corrupted token).

`tt-triage.txt` is absent on purpose: triage could not attach. It failed inside
`tt::umd::BlackholeTTDevice::wait_eth_core_training` / `TopologyDiscovery::get_connected_devices`,
i.e. it could not even complete Ethernet topology discovery, and wrote no LLM-output file.
`console.txt` is that failure with its full backtrace, which is the evidence.

Recovery followed $tt-device-usage's bounded sequence and succeeded: `tt-smi -ls --local` (8 boards)
-> `tt-smi -r` -> `tt-smi -ls --local` (8 boards) -> the mesh open/close smoke, which printed
MESH_SMOKE_OK. The interrupted evidence sweep was then resumed from its next step.
