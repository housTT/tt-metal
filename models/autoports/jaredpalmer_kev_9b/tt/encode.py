import re
from dataclasses import dataclass

from models.autoports.jaredpalmer_kev_9b.tt.api import SystemOneRequest, api_request, to_record

SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
MAX_STATE, MAX_BRANCH, MAX_PACKED = 384, 1024, 2048
SERVE_MAX_STATE = 65536
SERVE_MAX_BRANCH = SERVE_MAX_STATE + 8192
SERVE_MAX_PACKED = SERVE_MAX_STATE + SERVE_MAX_BRANCH
ROW_PASS_TOKENS = 16384


class ContextOverflow(ValueError):
    def __init__(self, message, state_tokens=None, max_state=None):
        super().__init__(message)
        self.state_tokens, self.max_state = state_tokens, max_state


_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def user_tokens(tok, text):
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def encode(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=False):
    state_tokens = user_tokens(tok, rec["state"])
    if strict and len(state_tokens) + 1 > max_state:
        raise ContextOverflow(
            f"state exceeds {max_state} tokens: {len(state_tokens) + 1}",
            state_tokens=len(state_tokens) + 1,
            max_state=max_state,
        )
    S = [tok.convert_tokens_to_ids(SPECIAL[0])] + state_tokens[: max_state - 1]
    ids, seg, pos = list(S), [0] * len(S), list(range(len(S)))
    q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL[1:])
    decide_idx, opt_idx = [], []
    for k, q in enumerate(rec["questions"], start=1):
        instr = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]
        br = instr + [t for sp in spans for t in sp] + [d_id]
        if len(br) > max_branch - len(S):
            raise ContextOverflow(
                f"branch too long: {len(br)} tokens with a {len(S)}-token state (row limit {max_branch})"
            )
        base = len(ids)
        p0 = len(S)
        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp)
            ends.append(cursor - 1)
        ids += br
        seg += [k] * len(br)
        pos += list(range(p0, p0 + len(br)))
        decide_idx.append(base + len(br) - 1)
        opt_idx.append([base + e for e in ends])
    return {
        "ids": ids,
        "seg": seg,
        "pos": pos,
        "decide_idx": decide_idx,
        "opt_idx": opt_idx,
        "state_tokens": len(state_tokens) + 1,
        "state_truncated": len(state_tokens) + 1 > max_state,
    }


def admit(tok, rec, truncate=False):
    try:
        return encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH, strict=not truncate)
    except ContextOverflow as e:
        if e.max_state is None:
            raise
        raise ContextOverflow(
            f"state is {e.state_tokens:,} tokens, over the {e.max_state:,}-token limit (the <state> token included): "
            "shorten the document or split it across requests",
            state_tokens=e.state_tokens,
            max_state=e.max_state,
        ) from None


def rows_of(enc):
    seg = enc["seg"]
    Ls = seg.count(0)
    rows, start = [], Ls
    for k, (d, oi) in enumerate(zip(enc["decide_idx"], enc["opt_idx"]), start=1):
        end = d + 1
        if seg[start] != k or seg[end - 1] != k:
            raise ValueError("branch layout mismatch")
        rows.append(
            {
                "ids": enc["ids"][start:end],
                "pos": enc["pos"][start:end],
                "decide": d - start,
                "opts": [o - start for o in oi],
            }
        )
        start = end
    return enc["ids"][:Ls], enc["pos"][:Ls], rows


@dataclass
class Row:
    qid: str
    qtype: str
    option_keys: list
    state_ids: list
    question_ids: list
    opt_positions: list
    decide_position: int
    legend: dict | None = None

    @property
    def ids(self):
        return self.state_ids + self.question_ids

    def __len__(self):
        return len(self.state_ids) + len(self.question_ids)


def rows_for_record(tok, rec, truncate=False):
    req = rec if isinstance(rec, SystemOneRequest) else SystemOneRequest.model_validate(api_request(rec))
    internal, meta = to_record(req)
    S, _, rows = rows_of(admit(tok, internal, truncate=truncate))
    return [
        Row(
            m["id"],
            m["type"],
            m["keys"],
            S,
            r["ids"],
            [len(S) + o for o in r["opts"]],
            len(S) + r["decide"],
            m.get("legend"),
        )
        for m, r in zip(meta, rows)
    ]
