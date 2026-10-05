import hashlib
import json
from dataclasses import dataclass

import torch

from models.autoports.cloudflare_clef.tt.encode import cache_key

HIDDEN = 5120
ALIGN = 128
MEDIA_TENSOR_KEYS = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")


@dataclass
class FakeHandle:
    S: int
    S0: int
    suffix_ids: torch.Tensor
    slot: int
    key: str = None


def media_summary(media):
    if not media:
        return None
    out = {}
    for key, value in media.items():
        if hasattr(value, "shape"):
            out[key] = list(value.shape)
        elif isinstance(value, list):
            out[key] = len(value)
        else:
            out[key] = value
    return out


def media_digest(media):
    if not media:
        return ""
    digest = hashlib.sha1()
    for key in MEDIA_TENSOR_KEYS:
        if key in media:
            digest.update(key.encode())
            digest.update(media[key].detach().cpu().contiguous().float().numpy().tobytes())
    return digest.hexdigest()


def seeded_rows(parts, n_rows, dim=HIDDEN):
    source = json.dumps(parts).encode()
    seed = int.from_bytes(hashlib.sha256(source).digest()[:8], "little") % (1 << 62)
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n_rows, dim, generator=generator)


class FakeEngine:
    supports_media = True

    def __init__(self, max_state_len=16384, max_tail_len=4096, snapshot_slots=1 << 30, dim=HIDDEN):
        self.max_state_len = max_state_len
        self.max_tail_len = max_tail_len
        self.max_len = max_state_len + max_tail_len
        self.snapshot_slots = snapshot_slots
        self.dim = dim
        self.handles = {}
        self.prefix_hidden = {}
        self.received = []
        self.precision = {"fake": True}
        self.timings = {"load_total_s": 0.0}

    def _record(self, method, token_ids, slot, media, **extra):
        self.received.append(
            dict(method=method, T=int(token_ids.shape[1]), slot=slot, media=media_summary(media), **extra)
        )

    def prefill_hidden(self, token_ids, slot=0, media=None):
        assert token_ids.shape[0] == 1
        T = token_ids.shape[1]
        assert T <= self.max_len, f"T={T} exceeds max_len={self.max_len}"
        self._record("prefill_hidden", token_ids, slot, media)
        self.handles.pop(slot, None)
        self.prefix_hidden.pop(slot, None)
        return seeded_rows(["full", token_ids[0].tolist(), media_digest(media)], T, self.dim)

    def prefill_state(self, state_ids, slot=0, key=None, media=None):
        assert state_ids.shape[0] == 1
        S = state_ids.shape[1]
        S0 = (S // ALIGN) * ALIGN
        assert S <= self.max_state_len, f"S={S} exceeds max_state_len={self.max_state_len}"
        assert 0 <= slot < self.snapshot_slots, f"slot {slot} out of range"
        self._record("prefill_state", state_ids, slot, media, key=key)
        ids = state_ids[0].tolist()
        key = key or cache_key(ids)
        self.prefix_hidden[slot] = seeded_rows(["prefix", ids[:S0], media_digest(media)], S0, self.dim)
        handle = FakeHandle(S, S0, state_ids[:, S0:S].clone(), slot, key)
        self.handles[slot] = handle
        return handle

    def schema_hidden(self, handle, schema_ids):
        if self.handles.get(handle.slot) is not handle:
            raise RuntimeError(f"slot {handle.slot} no longer holds state {handle.key[:8]}")
        tail = torch.cat([handle.suffix_ids, schema_ids], dim=1)
        L = tail.shape[1]
        assert handle.S0 + L <= self.max_len, f"S0+L={handle.S0 + L} exceeds max_len={self.max_len}"
        self._record("schema_hidden", tail, handle.slot, None, key=handle.key)
        return seeded_rows(["tail", handle.key, tail[0].tolist()], L, self.dim)
