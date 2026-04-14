"""Utility functions for STL formula evaluation."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import torch


def cond(
    pred: bool,
    true_fun: Callable[..., object],
    false_fun: Callable[..., object],
    *operands: object,
) -> object:
    """Conditional dispatch (non-JIT compatible branch)."""
    if pred:
        return true_fun(*operands)
    else:
        return false_fun(*operands)


def smooth_mask(
    T: int,
    t_start: float,
    t_end: float,
    scale: float,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Sigmoid-based smooth time mask for differentiable time boundaries."""
    xs = torch.arange(T, device=device).float()
    return torch.sigmoid(scale * (xs - t_start * T)) - torch.sigmoid(
        scale * (xs - t_end * T)
    )


def scan(
    f: Callable[[object, object], tuple[object, object]],
    init: object,
    xs: Optional[Sequence[object]] = None,
    length: Optional[int] = None,
) -> tuple[object, torch.Tensor]:
    """Sequential scan (fold with intermediate outputs)."""
    if xs is None:
        xs = [None] * (length or 0)
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, torch.stack(ys)


def convert_to_input_values(
    inputs: Union["Expression", torch.Tensor, tuple],  # noqa: F821
) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Convert Expression or Tensor inputs to their numerical values."""
    if not isinstance(inputs, tuple):
        # Avoid circular import — check by class name
        if hasattr(inputs, "value") and not isinstance(inputs, torch.Tensor):
            assert inputs.value is not None, "Expression does not have numerical values"
            return inputs.value
        elif isinstance(inputs, torch.Tensor):
            return inputs
        else:
            raise ValueError("Not a valid input trace")
    else:
        return (convert_to_input_values(inputs[0]), convert_to_input_values(inputs[1]))
