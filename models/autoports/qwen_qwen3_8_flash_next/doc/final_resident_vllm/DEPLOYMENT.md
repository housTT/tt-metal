# Deployment audit

Date: 2026-09-03

## Published package

- Hub repository: `tt-hous/qwen3.8-flash-next-p300x2`
- Visibility: private
- Hub revision: `5c390a73f86609fc7166ee9d78b0dd038c4b6662`
- OCI image digest:
  `sha256:c4805783cb377cf99c36c43a26f9cf41f4605732b8dd244d9a11211a9d290708`
- Manifest SHA256:
  `ab889d6a4d3bd07d532bf803429fd538b5d7c50dc9e04940d7da27ffb85255c2`
- Packaged code SHA256:
  `bc605f7e1f6f2342a207f91addbba6d3db84a7234ee93c549db91970b5add945`
- tt-metal commit: `60f1562e8ecf709bd778cb86c32dbfa5a06f8cb1`
  on `hous/qwen3.8-flash-next`, with the task's dirty tree packaged as-is
- tt-model commit: `5d36e7b6a2eb63d9972096db8a19ddeecb0cbf6b`
  on `hous/model-card-updates`
- vLLM TT plugin commit:
  `a48857ac68b17c31303e4809f348caaebbf10f74`, clean detached worktree
- Weights revision:
  `f5d08274bafd880402bd16f5e3e6c514136ec06c`

The local staged manifest and the manifest pulled from the published Hub
repository have the same SHA256. Docker reported the running consumer image ID
as the exact OCI image digest above.

## Validated lifecycle

The package completed its compile-time verification and exported a 2.0 GB OCI
layout. The exact staged digest then passed local serve, model discovery, and a
coherent completion before publication.

After publication, this consumer path was tested:

```bash
tt-model pull tt-hous/qwen3.8-flash-next-p300x2
tt-model serve tt-hous/qwen3.8-flash-next-p300x2 --local-only --follow
```

The pulled launch used image digest `c4805783...`, mesh `(4, 1)`, and devices
`0,1,2,3`. It reached API readiness in approximately 9 minutes 1 second with
cached OCI, Hugging Face, and TT kernel assets. Resident expert conversion is
not persisted, so all 48 layers are materialized again on every fresh process.

Consumer API checks:

- `GET /v1/models`: HTTP 200, correct model ID, `max_model_len=262144`;
- eight-token completion: HTTP 200 in 4.141 s, coherent text, exact usage
  accounting (`prompt=5`, `completion=8`, `total=13`);
- second completion: HTTP success, confirming reuse after the first request.

The validated consumer server was left running as
`tt-model-qwen3.8-flash-next-p300x2-default` on port 8000.

## Consumer runtime telemetry

After the first published-consumer completion, telemetry reported:

- 48 resident expert layers;
- 16,986,931,200 resident expert bytes per device;
- zero resident expert host-store bytes;
- zero expert H2D bytes and seconds;
- zero expert service seconds;
- zero route-read/device-stall seconds;
- PLE lookup active for 9 calls and 2,785,280 selected-row H2D bytes;
- all prohibited host-work flags false.

This proves that the published consumer path retains routed experts on TT and
keeps only the declared PLE n-gram row lookup/upload on the host.

