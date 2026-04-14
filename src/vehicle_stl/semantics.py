"""Pluggable semantics for STL formula evaluation.

Semantics define how logical connectives (conjunction, disjunction) are
interpreted as tensor reduction operations. Different semantics trade off
between exactness and differentiability:

- **Exact**: Uses true min/max. Correct but has zero gradients almost
  everywhere, which limits gradient-based optimization.
- **Softmax**: Smooth approximation via weighted softmax. Differentiable
  everywhere with controllable sharpness via temperature.
- **LogSumExp**: Smooth approximation via log-sum-exp. Differentiable
  everywhere, tends to over-approximate.

Users can define custom semantics by providing their own reduction functions
and identity elements.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol

import torch


class ReductionOp(Protocol):
    """A reduction operation over a tensor dimension.

    Must accept at minimum ``(signal, dim, keepdim)`` and may accept
    additional keyword arguments.
    """

    def __call__(
        self,
        signal: torch.Tensor,
        dim: int = 0,
        keepdim: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class Semantics:
    """Interpretation of logical connectives for STL evaluation.

    Attributes:
        conjunction: Reduction implementing 'and' — used by Always, And, and
            the prefix condition in Until.
        disjunction: Reduction implementing 'or' — used by Eventually, Or, and
            the existential search in Until.
        conjunction_identity: Identity element for conjunction masking. Masked
            positions are filled with this value so they do not affect the
            conjunction reduction (e.g., ``+1e9`` for min, ``0`` for sum).
        disjunction_identity: Identity element for disjunction masking
            (e.g., ``-1e9`` for max, ``1`` for product).
    """

    conjunction: ReductionOp
    disjunction: ReductionOp
    conjunction_identity: float
    disjunction_identity: float


# ---------------------------------------------------------------------------
# Built-in reduction functions
# ---------------------------------------------------------------------------


def _exact_min(
    signal: torch.Tensor, dim: int = 0, keepdim: bool = True, **kwargs: object
) -> torch.Tensor:
    return torch.min(signal, dim=dim, keepdim=keepdim)[0]


def _exact_max(
    signal: torch.Tensor, dim: int = 0, keepdim: bool = True, **kwargs: object
) -> torch.Tensor:
    return torch.max(signal, dim=dim, keepdim=keepdim)[0]


def _softmax_max(
    signal: torch.Tensor,
    dim: int = 0,
    keepdim: bool = True,
    temperature: float = 1.0,
    **kwargs: object,
) -> torch.Tensor:
    weights = torch.nn.functional.softmax(temperature * signal, dim=dim)
    return (weights * signal).sum(dim=dim, keepdim=keepdim)


def _softmax_min(
    signal: torch.Tensor,
    dim: int = 0,
    keepdim: bool = True,
    temperature: float = 1.0,
    **kwargs: object,
) -> torch.Tensor:
    return -_softmax_max(-signal, dim=dim, keepdim=keepdim, temperature=temperature)


def _logsumexp_max(
    signal: torch.Tensor,
    dim: int = 0,
    keepdim: bool = True,
    temperature: float = 1.0,
    **kwargs: object,
) -> torch.Tensor:
    return torch.logsumexp(temperature * signal, dim=dim, keepdim=keepdim) / temperature


def _logsumexp_min(
    signal: torch.Tensor,
    dim: int = 0,
    keepdim: bool = True,
    temperature: float = 1.0,
    **kwargs: object,
) -> torch.Tensor:
    return -_logsumexp_max(-signal, dim=dim, keepdim=keepdim, temperature=temperature)


# ---------------------------------------------------------------------------
# Pre-built semantics instances
# ---------------------------------------------------------------------------

LARGE_NUMBER: float = 1e9

EXACT = Semantics(
    conjunction=_exact_min,
    disjunction=_exact_max,
    conjunction_identity=LARGE_NUMBER,
    disjunction_identity=-LARGE_NUMBER,
)


def softmax(temperature: float = 1.0) -> Semantics:
    """Softmax-based smooth approximation semantics."""
    return Semantics(
        conjunction=partial(_softmax_min, temperature=temperature),
        disjunction=partial(_softmax_max, temperature=temperature),
        conjunction_identity=LARGE_NUMBER,
        disjunction_identity=-LARGE_NUMBER,
    )


def logsumexp(temperature: float = 1.0) -> Semantics:
    """LogSumExp-based smooth approximation semantics."""
    return Semantics(
        conjunction=partial(_logsumexp_min, temperature=temperature),
        disjunction=partial(_logsumexp_max, temperature=temperature),
        conjunction_identity=LARGE_NUMBER,
        disjunction_identity=-LARGE_NUMBER,
    )
