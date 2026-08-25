# Post-optimization evaluation artifact

`results.json` is the canonical, compact handoff for release tooling. `RESULTS.md` is generated from
the same data for human review. Neither file is a model card.

The artifact contains aggregate quality scores, test counts, serving checks, finalized performance,
exact source and weight revisions, configuration, limitations, fixed-subset document IDs, and
SHA-256 hashes of the raw evidence. It deliberately excludes prompts and model generations. Raw
lm-eval samples and JUnit XML remain outside the model tree.

## Inputs and generation

Run `run_api_gates.py` against the selected server, run the fixed IFEval and GPQA-Diamond subsets,
and save the hardware and host pytest results as JUnit XML. Then generate both outputs together:

- common lm-eval settings: `local-chat-completions`, concurrency 8, seed 42, temperature 0.6,
  top-k 20, top-p 0.95, no few-shot examples, and `--limit 0.05`;
- IFEval: document IDs 0–27, max model length 32,768, max generation 16,384;
- GPQA-Diamond: document IDs 0–9, max model length 65,536, max generation 32,768.

The builder refuses a different document-ID sequence. The exact task configuration and lm-eval,
Transformers, model, and tokenizer metadata are copied from the aggregate result files.

```bash
cd models/autoports/ornith_ai_ornith_1_0_35b

python doc/post_optimization_eval/run_api_gates.py \
  --base-url http://127.0.0.1:8100 \
  --output /tmp/ornith-api-gates.json

python doc/post_optimization_eval/build_artifact.py \
  --api-gates /tmp/ornith-api-gates.json \
  --ifeval-results /path/to/ifeval-results.json \
  --ifeval-samples /path/to/ifeval-samples.jsonl \
  --gpqa-results /path/to/gpqa-results.json \
  --gpqa-samples /path/to/gpqa-samples.jsonl \
  --latency-finalization /path/to/selected-sweep/FINALIZATION.json \
  --selected-config doc/datatype_sweep/selected_precision_config.json \
  --static-contract /tmp/ornith-production-contract.json \
  --device-junit /tmp/ornith-device-focused.junit.xml \
  --device-junit /tmp/ornith-device-full.junit.xml \
  --host-junit /tmp/ornith-host-tests.junit.xml \
  --tt-metal-root /path/to/tt-metal \
  --vllm-root /path/to/vllm \
  --output-json doc/post_optimization_eval/results.json \
  --output-markdown doc/post_optimization_eval/RESULTS.md
```

Repeat `--host-junit` for multiple host suites. Generation exits nonzero if an API, device, or host
gate failed. Quality rows are explicitly classified as measured-only because no matching GPU or
published reference is available; they do not affect the functional pass/fail status.
