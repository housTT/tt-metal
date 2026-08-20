"""Host-only reproduction of the TTI-release EngineCore deadlock.

Drives the real TTScheduler (async scheduling) against a faithful mock of the TT
model runner's cached-request bookkeeping (the plugin's
apply_cached_req_state_update + the runner's unconditional append of every
sampled token), with the benchmark's continuous-arrival client behaviour.
No hardware needed.
"""
import argparse
import random
import sys

sys.path.insert(0, "/home/ttuser/dev/ornith/vllm")
import vllm  # noqa
from vllm.platforms import current_platform

type(current_platform).check_and_update_config = classmethod(lambda c, cfg: None)

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm_tt_plugin.input_batch import apply_cached_req_state_update
from vllm_tt_plugin.scheduler import TTScheduler

import tests.v1.core.utils as U

U.AsyncScheduler = TTScheduler


class FakeRunner:
    def __init__(self, patched, log):
        self.requests = {}
        self.in_batch = {}
        self.num_tokens = {}
        self.patched = patched
        self.log = log
        self.drift_events = []
        self.empty_prefills = []

    def update_states(self, so):
        for rid in so.finished_req_ids:
            self.requests.pop(rid, None)
            self.in_batch.pop(rid, None)
            self.num_tokens.pop(rid, None)
        scheduled = so.num_scheduled_tokens.keys()
        for rid in list(self.in_batch.keys() - scheduled):
            del self.in_batch[rid]
        to_add = []
        for new in so.scheduled_new_reqs:
            self.requests[new.req_id] = CachedRequestState(
                req_id=new.req_id,
                prompt_token_ids=new.prompt_token_ids,
                mm_features=new.mm_features,
                sampling_params=new.sampling_params,
                pooling_params=new.pooling_params,
                generator=None,
                block_ids=new.block_ids,
                num_computed_tokens=new.num_computed_tokens,
                output_token_ids=[],
            )
            to_add.append(new.req_id)
        rd = so.scheduled_cached_reqs
        for i, rid in enumerate(rd.req_ids):
            st = self.requests[rid]
            resumed = rid in rd.resumed_req_ids
            idx = self.in_batch.get(rid)
            if resumed:
                drift = len(st.output_token_ids) - rd.num_output_tokens[i]
                self.drift_events.append((rid, len(st.output_token_ids), rd.num_output_tokens[i], drift))
                self.log(
                    f"  resume {rid}: runner_out={len(st.output_token_ids)} sched_out={rd.num_output_tokens[i]} drift={drift}"
                )
            if self.patched:
                changed = apply_cached_req_state_update(
                    st,
                    rd.num_computed_tokens[i],
                    rd.new_block_ids[i],
                    resumed,
                    num_output_tokens=rd.num_output_tokens[i],
                    all_token_ids=rd.all_token_ids.get(rid),
                    in_persistent_batch=idx is not None,
                    async_scheduling=True,
                )
            else:
                changed = apply_cached_req_state_update(st, rd.num_computed_tokens[i], rd.new_block_ids[i], resumed)
            if idx is None:
                to_add.append(rid)
                continue
            if changed:
                self.num_tokens[rid] = len(st.prompt_token_ids) + rd.num_output_tokens[i]
        for rid in to_add:
            st = self.requests[rid]
            self.in_batch[rid] = len(self.in_batch)
            self.num_tokens[rid] = len(st.prompt_token_ids) + len(st.output_token_ids)

    def run(self, so, step):
        req_ids, sampled = [], []
        for rid, n_sched in so.num_scheduled_tokens.items():
            st = self.requests[rid]
            n_prompt = len(st.prompt_token_ids)
            is_prefill = st.num_computed_tokens < n_prompt
            req_ids.append(rid)
            if is_prefill:
                prompt_lens = st.num_computed_tokens + n_sched
                if prompt_lens < self.num_tokens[rid]:
                    self.log(
                        f"  prefill {rid}: prompt_lens={prompt_lens} < num_tokens={self.num_tokens[rid]} -> INTERMEDIATE, no token"
                    )
                    self.empty_prefills.append((step, rid, prompt_lens, self.num_tokens[rid]))
                    sampled.append([])
                    continue
            tok = 1000 + len(st.output_token_ids)
            st.output_token_ids.append(tok)
            self.num_tokens[rid] += 1
            sampled.append([tok])
        return ModelRunnerOutput(
            req_ids=req_ids, req_id_to_index={r: i for i, r in enumerate(req_ids)}, sampled_token_ids=sampled
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patched", action="store_true")
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--prompt", type=int, default=320)
    ap.add_argument("--prompt-jitter", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=18)
    ap.add_argument("--num-blocks", type=int, default=127)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    def log(msg):
        if args.verbose:
            print(msg)

    sched = U.create_scheduler(
        model="/tmp/fake_tt_qwen3",
        skip_tokenizer_init=True,
        async_scheduling=True,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=False,
        max_num_seqs=args.max_num_seqs,
    )
    assert isinstance(sched, TTScheduler)

    lens = [
        args.prompt + (rng.randint(0, args.prompt_jitter) if args.prompt_jitter else 0) for _ in range(args.num_prompts)
    ]
    pool = []
    for i, L in enumerate(lens):
        r = U.create_requests(
            num_requests=1, num_tokens=L, max_tokens=args.max_tokens, block_size=args.block_size, req_ids=[f"r{i}"]
        )[0]
        pool.append(r)
    pending = list(pool)
    inflight = 0
    for _ in range(min(args.concurrency, len(pending))):
        sched.add_request(pending.pop(0))
        inflight += 1

    runner = FakeRunner(args.patched, log)
    prev = None
    idle = 0
    done = 0
    for step in range(args.max_steps):
        so = sched.schedule()
        if so.total_num_scheduled_tokens == 0 and sched.has_unfinished_requests():
            idle += 1
            if idle > 30:
                stuck = [
                    (
                        r.request_id,
                        r.num_computed_tokens,
                        r.num_tokens,
                        r.num_output_placeholders,
                        r.num_output_tokens,
                        r.max_tokens,
                        r.num_prompt_tokens,
                    )
                    for r in sched.running
                ]
                print(
                    f"RESULT: HANG at step {step} after {done} completed; running={stuck}; "
                    f"drift_events={runner.drift_events}; empty_prefills={runner.empty_prefills}"
                )
                return 1
        else:
            idle = 0
        runner.update_states(so)
        out = runner.run(so, step)
        if prev is not None:
            eco = sched.update_from_output(prev[0], prev[1])
            for o in eco.values():
                for x in o.outputs:
                    if x.finish_reason is not None:
                        done += 1
                        inflight -= 1
                        if pending:
                            sched.add_request(pending.pop(0))
                            inflight += 1
        prev = (so, out)
        if (
            not sched.has_unfinished_requests()
            and not pending
            and prev is not None
            and so.total_num_scheduled_tokens == 0
        ):
            sched.update_from_output(prev[0], prev[1])
            break
    print(
        f"RESULT: COMPLETED {done}/{args.num_prompts} in {step} steps; "
        f"drift_events={len(runner.drift_events)}; empty_prefills={len(runner.empty_prefills)}"
    )
    return 0


sys.exit(main())
