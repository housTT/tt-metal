# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Explicit experimental policy selection; runtime defaults do not read environment."""

import json
import os

from ..tt.optimized_decoder import DecoderConfig, OptimizedDecoder, PrecisionPolicy


def selected_candidate():
    from .optimization_composite_candidates import (
        ChunkL1AdaptedCandidate,
        CombinedOutputKDAPrefillCandidate,
        CombinedOutputPrefillCandidate,
        FlatOutputKDAPrefillCandidate,
        FlatOutputPrefillCandidate,
        NativeDeltaDecodeCandidate,
        NativeFlatPrefillCandidate,
        PrefillChunkComputeCandidate,
        PrefillInputL1Candidate,
        RecurrentOuterBroadcastCandidate,
        SeparateMLPKDAPrefillCandidate,
        SeparatePrefillMLPCandidate,
        TiledHeadSplitCandidate,
    )
    from .optimization_extra_candidates import (
        ActivationCandidate,
        CacheSDPACandidate,
        CombinedDecodeCandidate,
        CombinedKDAPrefillCandidate,
        CombinedOutputCandidate,
        CombinedPrefillCandidate,
        KDAConvCandidate,
        LargePrefillCandidate,
        LowerMovementCandidate,
        PackedAlignedMLPCandidate,
        ProjectionOutputCandidate,
        ProjectionTopologyCandidate,
        RecurrentConfigCandidate,
        ShardedHeadNormCandidate,
        SharedMLPInputCandidate,
        TiledRotaryIndicesCandidate,
    )
    from .optimization_final_candidates import (
        FinalCacheCandidate,
        FinalLargePrefillMLPBlockCandidate,
        FinalPackedMLPCandidate,
        FinalPrefillGridCandidate,
        FinalPrefillMLPBlockCandidate,
        FinalTopologyCandidate,
        FinalZApproxCandidate,
    )

    base = {
        "default": OptimizedDecoder,
        "final_cache": FinalCacheCandidate,
        "final_prefill_mlp_grid": FinalLargePrefillMLPBlockCandidate,
        "final_z_approx": FinalZApproxCandidate,
        "final_packed": FinalPackedMLPCandidate,
        "final_topology": FinalTopologyCandidate,
        "final_prefill_mlp": FinalPrefillMLPBlockCandidate,
        "final_prefill_grid": FinalPrefillGridCandidate,
        "recurrent": RecurrentConfigCandidate,
        "tiled_rope": TiledRotaryIndicesCandidate,
        "large_prefill": LargePrefillCandidate,
        "cache_sdpa": CacheSDPACandidate,
        "topology": ProjectionTopologyCandidate,
        "activation": ActivationCandidate,
        "sharded_head_norm": ShardedHeadNormCandidate,
        "shared_mlp_input": SharedMLPInputCandidate,
        "kda_conv": KDAConvCandidate,
        "packed_aligned": PackedAlignedMLPCandidate,
        "combined_decode": CombinedDecodeCandidate,
        "combined_prefill": CombinedPrefillCandidate,
        "combined_kda": CombinedKDAPrefillCandidate,
        "lower_movement": LowerMovementCandidate,
        "projection_output": ProjectionOutputCandidate,
        "combined_output": CombinedOutputCandidate,
        "native_delta": NativeDeltaDecodeCandidate,
        "chunk_compute": PrefillChunkComputeCandidate,
        "output_prefill": CombinedOutputPrefillCandidate,
        "output_kda_prefill": CombinedOutputKDAPrefillCandidate,
        "prefill_input_l1": PrefillInputL1Candidate,
        "outer_broadcast": RecurrentOuterBroadcastCandidate,
        "tiled_head_split": TiledHeadSplitCandidate,
        "native_flat_prefill": NativeFlatPrefillCandidate,
        "chunk_l1_adapted": ChunkL1AdaptedCandidate,
        "flat_output_prefill": FlatOutputPrefillCandidate,
        "flat_output_kda_prefill": FlatOutputKDAPrefillCandidate,
        "separate_prefill_mlp": SeparatePrefillMLPCandidate,
        "separate_mlp_kda": SeparateMLPKDAPrefillCandidate,
    }[os.environ.get("ORNITH_OPT_VARIANT", "default")]
    variant = os.environ.get("ORNITH_OPT_VARIANT", "default")
    if variant == "default" or variant.startswith("final_"):
        policy_type, config_type = PrecisionPolicy, DecoderConfig
    else:
        from .optimization_baseline import DecoderConfig as config_type
        from .optimization_baseline import PrecisionPolicy as policy_type

    class Candidate(base):
        @classmethod
        def from_state_dict(cls, state_dict, **kwargs):
            policy = policy_type(**json.loads(os.environ.get("ORNITH_OPT_POLICY", "{}")))
            config = config_type(**json.loads(os.environ.get("ORNITH_OPT_CONFIG", "{}")))
            return super().from_state_dict(state_dict, policy=policy, config=config, **kwargs)

    return Candidate
