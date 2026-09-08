# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Execute the shared penalty function with CPU operation boundaries only."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[4]


def definitions(path, names, namespace):
    nodes = [
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)


class CpuTensor:
    def __init__(self, value):
        self.value = value

    def deallocate(self):
        pass


def raw(value):
    return value.value if isinstance(value, CpuTensor) else value


def binary(op, left, right, *, output_tensor=None, **kwargs):
    result = op(raw(left), raw(right))
    if output_tensor is not None:
        output_tensor.value.copy_(result)
        return output_tensor
    return CpuTensor(result)


def common_penalties(logits, prompt_mask, output_counts, presence, frequency, repetition):
    ttnn = SimpleNamespace(
        bfloat16=torch.bfloat16,
        typecast=lambda value, dtype, **kwargs: CpuTensor(raw(value).to(dtype)),
        multiply=lambda a, b, **kwargs: binary(torch.mul, a, b, **kwargs),
        subtract=lambda a, b, **kwargs: binary(torch.sub, a, b, **kwargs),
        add=lambda a, b, **kwargs: binary(torch.add, a, b, **kwargs),
        gt=lambda a, b, **kwargs: binary(torch.gt, a, b, **kwargs),
        where=lambda mask, a, b, **kwargs: CpuTensor(torch.where(raw(mask).bool(), raw(a), raw(b))),
    )
    namespace = {"ttnn": ttnn}
    definitions(REPO / "models/common/sampling/tt_penalties.py", {"apply_penalties"}, namespace)
    context = SimpleNamespace(
        prompt_mask=CpuTensor(prompt_mask.to(torch.int32)),
        output_mask=CpuTensor((output_counts > 0).to(torch.int32)),
        output_counts=CpuTensor(output_counts.to(torch.int32)),
        presence_penalties=CpuTensor(presence[:, None].to(torch.bfloat16)),
        frequency_penalties=CpuTensor(frequency[:, None].to(torch.bfloat16)),
        repetition_penalties=CpuTensor(repetition[:, None].to(torch.bfloat16)),
        inverse_repetition_penalties=CpuTensor((1 / repetition[:, None]).to(torch.bfloat16)),
        sub_core_grids=None,
    )
    target = CpuTensor(logits.clone().to(torch.bfloat16))
    result = namespace["apply_penalties"](target, context)
    assert result is target, "Penalty updates must retain their trace-bound logits buffer"
    return result.value.float()


def reference_penalties(logits, prompt_mask, output_counts, presence, frequency, repetition):
    repeated = prompt_mask | (output_counts > 0)
    factor = torch.where(repeated, repetition[:, None], 1.0)
    result = torch.where(logits > 0, logits / factor, logits * factor)
    result = result - frequency[:, None] * output_counts
    return result - presence[:, None] * (output_counts > 0)


@pytest.mark.parametrize(
    "presence,frequency,repetition",
    [
        (0.0, 0.0, 1.0),
        (2.0, 0.0, 1.0),
        (-1.5, 0.0, 1.0),
        (0.0, 0.5, 1.0),
        (0.0, -0.5, 1.0),
        (0.0, 0.0, 2.0),
        (0.0, 0.0, 0.5),
        (2.0, 0.0, 2.0),
        (0.0, 0.5, 2.0),
        (2.0, 0.5, 2.0),
        (-1.5, -0.5, 2.0),
        (2.0, 0.5, 0.5),
    ],
)
def test_common_penalties_follow_host_order_and_history_scope(presence, frequency, repetition):
    # Every sign, including additive penalties crossing zero; columns distinguish
    # generated once, generated repeatedly, prompt only, both histories, unseen.
    logits = torch.tensor([[4.0, -4.0, 0.5, -0.5, 0.0], [-0.5, 0.5, -4.0, 4.0, 0.5]])
    prompt = torch.tensor([[False, False, True, True, False]] * 2)
    counts = torch.tensor([[1, 3, 0, 2, 0]] * 2)
    params = [torch.full((2,), value) for value in (presence, frequency, repetition)]
    actual = common_penalties(logits, prompt, counts, *params)
    expected = reference_penalties(logits, prompt, counts, *params)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_combined_penalties_change_winner_at_the_host_boundary():
    logits = torch.tensor([[4.0, 0.5]])
    prompt, counts = torch.zeros(1, 2, dtype=torch.bool), torch.tensor([[1, 0]])
    actual = common_penalties(logits, prompt, counts, torch.tensor([2.0]), torch.tensor([0.0]), torch.tensor([2.0]))
    torch.testing.assert_close(actual, torch.tensor([[0.0, 0.5]]), rtol=0, atol=0)
    assert actual.argmax(-1).tolist() == [1]
