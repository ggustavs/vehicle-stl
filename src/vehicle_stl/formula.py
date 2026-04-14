"""Signal Temporal Logic formula classes with pluggable semantics.

Forked from stlcg++ (https://github.com/UW-CTRL/stlcg-plus-plus) by
Karen Leung et al., with the following changes:

- Conjunction/disjunction reductions are parameterised via a ``Semantics``
  object instead of being dispatched through ``approx_method`` strings.
- All formula classes accept an optional ``semantics`` argument (defaults
  to ``EXACT``).
- Code quality improvements (removed nested class in Until, explicit
  imports, type annotations).
- ``Expression``, ``Predicate``, and visualization support are preserved
  for standalone and falsification use.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import torch

from .semantics import EXACT, Semantics
from .utils import cond, convert_to_input_values, scan, smooth_mask

# ==========================================================================
# Expressions
# ==========================================================================


class Expression(torch.nn.Module):
    """A named symbolic value that participates in formula evaluation."""

    def __init__(self, name: str, value: object = None) -> None:
        super().__init__()
        self.value = value
        self.name = name

    def set_name(self, new_name: str) -> None:
        self.name = new_name

    def set_value(self, new_value: object) -> None:
        self.value = new_value

    def get_name(self) -> str:
        return self.name

    def forward(self) -> object:
        return self.value

    # -- arithmetic --

    def __neg__(self) -> Expression:
        return Expression("-" + self.name, -self.value)

    def __add__(self, other: Expression) -> Expression:
        if isinstance(other, Expression):
            return Expression(
                self.name + " + " + other.name, self.value + other.value
            )
        return Expression(self.name + " + other", self.value + other)

    def __radd__(self, other: object) -> Expression:
        return self.__add__(other)  # type: ignore[arg-type]

    def __sub__(self, other: Expression) -> Expression:
        if isinstance(other, Expression):
            return Expression(
                self.name + " - " + other.name, self.value - other.value
            )
        return Expression(self.name + " - other", self.value - other)

    def __rsub__(self, other: object) -> Expression:
        return Expression("other - " + self.name, other - self.value)

    def __mul__(self, other: Expression) -> Expression:
        if isinstance(other, Expression):
            return Expression(
                self.name + " x " + other.name, self.value * other.value
            )
        return Expression(self.name + " x other", self.value * other)

    def __rmul__(self, other: object) -> Expression:
        return self.__mul__(other)  # type: ignore[arg-type]

    def __truediv__(self, other: Expression) -> Expression:
        if isinstance(other, Expression):
            return Expression(
                self.name + " / " + other.name, self.value / other.value
            )
        return Expression(self.name + " / other", self.value / other)

    def __str__(self) -> str:
        return str(self.name)


# ==========================================================================
# Predicates
# ==========================================================================


class Predicate(torch.nn.Module):
    """A named function applied to a signal before formula evaluation."""

    def __init__(self, name: str, predicate_function: Callable[..., torch.Tensor]) -> None:
        super().__init__()
        self.name = name
        self.predicate_function = predicate_function

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.predicate_function(signal)

    def set_name(self, new_name: str) -> None:
        self.name = new_name

    # -- arithmetic (produce new Predicates) --

    def __neg__(self) -> Predicate:
        return Predicate("- " + self.name, lambda x: -self.predicate_function(x))

    def __add__(self, other: Predicate) -> Predicate:
        if not isinstance(other, Predicate):
            raise ValueError("Type error. Must be Predicate")
        return Predicate(
            self.name + " + " + other.name,
            lambda x: self.predicate_function(x) + other.predicate_function(x),
        )

    def __radd__(self, other: Predicate) -> Predicate:
        return self.__add__(other)

    def __sub__(self, other: Predicate) -> Predicate:
        if not isinstance(other, Predicate):
            raise ValueError("Type error. Must be Predicate")
        return Predicate(
            self.name + " - " + other.name,
            lambda x: self.predicate_function(x) - other.predicate_function(x),
        )

    def __rsub__(self, other: Predicate) -> Predicate:
        return self.__sub__(other)

    def __mul__(self, other: Predicate) -> Predicate:
        if not isinstance(other, Predicate):
            raise ValueError("Type error. Must be Predicate")
        return Predicate(
            self.name + " x " + other.name,
            lambda x: self.predicate_function(x) * other.predicate_function(x),
        )

    def __rmul__(self, other: Predicate) -> Predicate:
        return self.__mul__(other)

    def __truediv__(self, other: Predicate) -> Predicate:
        if not isinstance(other, Predicate):
            raise ValueError("Type error. Must be Predicate")
        return Predicate(
            self.name + " / " + other.name,
            lambda x: self.predicate_function(x) / other.predicate_function(x),
        )

    # -- comparators (produce STL formulas) --

    def __lt__(self, rhs: Union[Predicate, float, torch.Tensor]) -> LessThan:
        return LessThan(self, rhs)

    def __le__(self, rhs: Union[Predicate, float, torch.Tensor]) -> LessThan:
        return LessThan(self, rhs)

    def __gt__(self, rhs: Union[Predicate, float, torch.Tensor]) -> GreaterThan:
        return GreaterThan(self, rhs)

    def __ge__(self, rhs: Union[Predicate, float, torch.Tensor]) -> GreaterThan:
        return GreaterThan(self, rhs)

    def __eq__(self, rhs: Union[Predicate, float, torch.Tensor]) -> Equal:  # type: ignore[override]
        return Equal(self, rhs)

    def __hash__(self) -> int:
        return hash((self.name, self.predicate_function))

    def __str__(self) -> str:
        return str(self.name)


# ==========================================================================
# STL Formula base class
# ==========================================================================


class STLFormula(torch.nn.Module):
    """Base class for all STL formula nodes.

    If ``Expressions`` and ``Predicates`` are used, signals are converted
    automatically via :func:`convert_to_input_values`.  Otherwise the caller
    is responsible for providing plain tensors.
    """

    def __init__(self, semantics: Optional[Semantics] = None) -> None:
        super().__init__()
        self.semantics: Semantics = semantics or EXACT

    def robustness_trace(self, signal: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Compute the robustness trace for the given signal.

        Returns a tensor of shape ``[time_dim, ...]`` with one robustness
        value per timestep.
        """
        raise NotImplementedError("robustness_trace not yet implemented")

    def robustness(self, signal: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Scalar robustness (first element of the trace)."""
        return self.forward(signal, **kwargs)[0]

    def eval_trace(self, signal: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Boolean trace (``robustness_trace > 0``)."""
        return self.forward(signal, **kwargs) > 0

    def eval(self, signal: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Boolean robustness (``robustness > 0``)."""
        return self.robustness(signal, **kwargs) > 0

    def forward(self, signal: torch.Tensor, **kwargs: object) -> torch.Tensor:
        inputs = convert_to_input_values(signal)
        return self.robustness_trace(inputs, **kwargs)

    def _next_function(self) -> list[object]:
        """Subformula list for visualization."""
        raise NotImplementedError("_next_function not yet implemented")

    # -- logical combinators --

    def __and__(self, psi: STLFormula) -> And:
        return And(self, psi)

    def __or__(self, psi: STLFormula) -> Or:
        return Or(self, psi)

    def __invert__(self) -> Negation:
        return Negation(self)


# ==========================================================================
# Identity
# ==========================================================================


class Identity(STLFormula):
    """Pass-through formula — returns the input signal unchanged."""

    def __init__(self, name: str = "x", semantics: Optional[Semantics] = None) -> None:
        super().__init__(semantics=semantics)
        self.name = name

    def robustness_trace(self, signal: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return signal

    def _next_function(self) -> list[object]:
        return []

    def __str__(self) -> str:
        return self.name


# ==========================================================================
# Comparators
# ==========================================================================


class LessThan(STLFormula):
    """Robustness: ``rhs - lhs(signal)``."""

    def __init__(
        self,
        lhs: Union[Predicate, Expression, str],
        rhs: Union[float, torch.Tensor],
        semantics: Optional[Semantics] = None,
    ) -> None:
        super().__init__(semantics=semantics)
        self.lhs = lhs
        self.rhs = rhs

    def robustness_trace(self, signal: Union[torch.Tensor, Expression], **kwargs: object) -> torch.Tensor:
        if isinstance(self.lhs, Predicate):
            return self.rhs - self.lhs(signal)
        elif isinstance(signal, Expression):
            assert signal.value is not None, "Expression does not have numerical values"
            return self.rhs - signal.value
        else:
            return self.rhs - signal

    def _next_function(self) -> list[object]:
        return [self.lhs, self.rhs]

    def __str__(self) -> str:
        lhs_str = self.lhs
        if isinstance(self.lhs, (Predicate, Expression)):
            lhs_str = self.lhs.name
        return str(lhs_str) + " < " + str(self.rhs)


class GreaterThan(STLFormula):
    """Robustness: ``lhs(signal) - rhs``."""

    def __init__(
        self,
        lhs: Union[Predicate, Expression, str],
        rhs: Union[float, torch.Tensor],
        semantics: Optional[Semantics] = None,
    ) -> None:
        super().__init__(semantics=semantics)
        self.lhs = lhs
        self.rhs = rhs

    def robustness_trace(self, signal: Union[torch.Tensor, Expression], **kwargs: object) -> torch.Tensor:
        if isinstance(self.lhs, Predicate):
            return self.lhs(signal) - self.rhs
        elif isinstance(signal, Expression):
            assert signal.value is not None, "Expression does not have numerical values"
            return signal.value - self.rhs
        else:
            return signal - self.rhs

    def _next_function(self) -> list[object]:
        return [self.lhs, self.rhs]

    def __str__(self) -> str:
        lhs_str = self.lhs
        if isinstance(self.lhs, (Predicate, Expression)):
            lhs_str = self.lhs.name
        return str(lhs_str) + " > " + str(self.rhs)


class Equal(STLFormula):
    """Robustness: ``-|lhs(signal) - rhs|``."""

    def __init__(
        self,
        lhs: Union[Predicate, Expression, str],
        rhs: Union[float, torch.Tensor],
        semantics: Optional[Semantics] = None,
    ) -> None:
        super().__init__(semantics=semantics)
        self.lhs = lhs
        self.rhs = rhs

    def robustness_trace(self, signal: Union[torch.Tensor, Expression], **kwargs: object) -> torch.Tensor:
        if isinstance(self.lhs, Predicate):
            return -torch.abs(self.lhs(signal) - self.rhs)
        elif isinstance(signal, Expression):
            assert signal.value is not None, "Expression does not have numerical values"
            return -torch.abs(signal.value - self.rhs)
        else:
            return -torch.abs(signal - self.rhs)

    def _next_function(self) -> list[object]:
        return [self.lhs, self.rhs]

    def __str__(self) -> str:
        lhs_str = self.lhs
        if isinstance(self.lhs, (Predicate, Expression)):
            lhs_str = self.lhs.name
        return str(lhs_str) + " == " + str(self.rhs)


# ==========================================================================
# Propositional connectives
# ==========================================================================


def _separate_and(
    formula: STLFormula, signal: object, **kwargs: object
) -> torch.Tensor:
    """Recursively flatten nested And nodes for parallel reduction."""
    if not isinstance(formula, And):
        return formula(signal, **kwargs).unsqueeze(-1)
    if isinstance(signal, tuple):
        return torch.cat(
            [
                _separate_and(formula.subformula1, signal[0], **kwargs),
                _separate_and(formula.subformula2, signal[1], **kwargs),
            ],
            dim=-1,
        )
    return torch.cat(
        [
            _separate_and(formula.subformula1, signal, **kwargs),
            _separate_and(formula.subformula2, signal, **kwargs),
        ],
        dim=-1,
    )


def _separate_or(
    formula: STLFormula, signal: object, **kwargs: object
) -> torch.Tensor:
    """Recursively flatten nested Or nodes for parallel reduction."""
    if not isinstance(formula, Or):
        return formula(signal, **kwargs).unsqueeze(-1)
    if isinstance(signal, tuple):
        return torch.cat(
            [
                _separate_or(formula.subformula1, signal[0], **kwargs),
                _separate_or(formula.subformula2, signal[1], **kwargs),
            ],
            dim=-1,
        )
    return torch.cat(
        [
            _separate_or(formula.subformula1, signal, **kwargs),
            _separate_or(formula.subformula2, signal, **kwargs),
        ],
        dim=-1,
    )


class Negation(STLFormula):
    """Negation: ``-subformula(signal)``."""

    def __init__(self, subformula: STLFormula, semantics: Optional[Semantics] = None) -> None:
        super().__init__(semantics=semantics)
        self.subformula = subformula

    def robustness_trace(self, signal: Union[torch.Tensor, Expression], **kwargs: object) -> torch.Tensor:
        return -self.subformula(signal, **kwargs)

    def _next_function(self) -> list[object]:
        return [self.subformula]

    def __str__(self) -> str:
        return "\u00ac(" + str(self.subformula) + ")"


class And(STLFormula):
    """Conjunction: ``semantics.conjunction(subformula1, subformula2)``."""

    def __init__(
        self,
        subformula1: STLFormula,
        subformula2: STLFormula,
        semantics: Optional[Semantics] = None,
    ) -> None:
        super().__init__(semantics=semantics)
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def robustness_trace(self, inputs: object, **kwargs: object) -> torch.Tensor:
        xx = _separate_and(self, inputs, **kwargs)
        return self.semantics.conjunction(xx, dim=-1, keepdim=False)

    def _next_function(self) -> list[object]:
        return [self.subformula1, self.subformula2]

    def __str__(self) -> str:
        return "(" + str(self.subformula1) + ") \u2227 (" + str(self.subformula2) + ")"


class Or(STLFormula):
    """Disjunction: ``semantics.disjunction(subformula1, subformula2)``."""

    def __init__(
        self,
        subformula1: STLFormula,
        subformula2: STLFormula,
        semantics: Optional[Semantics] = None,
    ) -> None:
        super().__init__(semantics=semantics)
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def robustness_trace(self, inputs: object, **kwargs: object) -> torch.Tensor:
        xx = _separate_or(self, inputs, **kwargs)
        return self.semantics.disjunction(xx, dim=-1, keepdim=False)

    def _next_function(self) -> list[object]:
        return [self.subformula1, self.subformula2]

    def __str__(self) -> str:
        return "(" + str(self.subformula1) + ") \u2228 (" + str(self.subformula2) + ")"


class Implies(STLFormula):
    """Implication: ``disjunction(-subformula1, subformula2)``."""

    def __init__(
        self,
        subformula1: STLFormula,
        subformula2: STLFormula,
        semantics: Optional[Semantics] = None,
    ) -> None:
        super().__init__(semantics=semantics)
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def robustness_trace(self, trace: object, **kwargs: object) -> torch.Tensor:
        if isinstance(trace, tuple):
            trace1, trace2 = trace
            signal1 = self.subformula1(trace1, **kwargs)
            signal2 = self.subformula2(trace2, **kwargs)
        else:
            signal1 = self.subformula1(trace, **kwargs)
            signal2 = self.subformula2(trace, **kwargs)
        xx = torch.stack([-signal1, signal2], dim=-1)
        return self.semantics.disjunction(xx, dim=-1, keepdim=False)

    def _next_function(self) -> list[object]:
        return [self.subformula1, self.subformula2]

    def __str__(self) -> str:
        return "(" + str(self.subformula1) + ") \u21d2 (" + str(self.subformula2) + ")"


# ==========================================================================
# Standard temporal operators (matrix-based, O(T^2) memory)
# ==========================================================================


class Eventually(STLFormula):
    """Eventually (Finally / Diamond): property holds at some point in the interval."""

    def __init__(
        self,
        subformula: Optional[STLFormula] = None,
        interval: Optional[list[Union[int, float]]] = None,
        semantics: Optional[Semantics] = None,
    ) -> None:
        super().__init__(semantics=semantics)
        self.interval = interval
        self.subformula = subformula or Identity()
        self._interval = [0, torch.inf] if self.interval is None else self.interval

    def robustness_trace(
        self,
        signal: torch.Tensor,
        padding: Optional[str] = None,
        large_number: float = 1e9,
        **kwargs: object,
    ) -> torch.Tensor:
        device = signal.device
        time_dim = 0
        signal = self.subformula(signal, padding=padding, large_number=large_number, **kwargs)
        T = signal.shape[time_dim]
        mask_value = self.semantics.disjunction_identity

        def true_func(_interval: list, T: int) -> tuple:
            return [_interval[0], T - 1], -1.0, _interval[0]

        def false_func(_interval: list, T: int) -> tuple:
            return _interval, 1.0, 0

        interval, _, offset = cond(
            self._interval[1] == torch.inf, true_func, false_func, self._interval, T
        )

        signal_matrix = signal.reshape([T, 1]) @ torch.ones([1, T], device=device)
        if padding == "last":
            pad_value = signal[-1]
        elif padding == "mean":
            pad_value = signal.mean(time_dim)
        else:
            pad_value = self.semantics.disjunction_identity

        signal_pad = torch.ones([interval[1] + 1, T], device=device) * pad_value
        signal_padded = torch.cat([signal_matrix, signal_pad], dim=time_dim)
        subsignal_mask = torch.tril(torch.ones([T + interval[1] + 1, T], device=device))
        time_interval_mask = torch.triu(
            torch.ones([T + interval[1] + 1, T], device=device), -interval[-1] - offset
        ) * torch.tril(
            torch.ones([T + interval[1] + 1, T], device=device), -interval[0]
        )
        masked_signal_matrix = torch.where(
            (time_interval_mask * subsignal_mask) == 1.0, signal_padded, mask_value
        )
        return self.semantics.disjunction(masked_signal_matrix, dim=time_dim, keepdim=False)

    def _next_function(self) -> list[object]:
        return [self.subformula]

    def __str__(self) -> str:
        return "\u2662 " + str(self._interval) + "( " + str(self.subformula) + " )"


class Always(STLFormula):
    """Always (Globally / Box): property holds at every point in the interval."""

    def __init__(
        self,
        subformula: Optional[STLFormula] = None,
        interval: Optional[list[Union[int, float]]] = None,
        semantics: Optional[Semantics] = None,
    ) -> None:
        super().__init__(semantics=semantics)
        self.interval = interval
        self.subformula = subformula or Identity()
        self._interval = [0, torch.inf] if self.interval is None else self.interval

    def robustness_trace(
        self,
        signal: torch.Tensor,
        padding: Optional[str] = None,
        large_number: float = 1e9,
        **kwargs: object,
    ) -> torch.Tensor:
        device = signal.device
        time_dim = 0
        signal = self.subformula(signal, padding=padding, large_number=large_number, **kwargs)
        T = signal.shape[time_dim]
        mask_value = self.semantics.conjunction_identity

        def true_func(_interval: list, T: int) -> tuple:
            return [_interval[0], T - 1], -1.0, _interval[0]

        def false_func(_interval: list, T: int) -> tuple:
            return _interval, 1.0, 0

        interval, sign, offset = cond(
            self._interval[1] == torch.inf, true_func, false_func, self._interval, T
        )

        signal_matrix = signal.reshape([T, 1]) @ torch.ones([1, T], device=device)
        if padding == "last":
            pad_value = signal[-1]
        elif padding == "mean":
            pad_value = signal.mean(time_dim)
        else:
            pad_value = -large_number

        signal_pad = torch.cat(
            [
                torch.ones([interval[1], T], device=device) * sign * pad_value,
                torch.ones([1, T], device=device) * pad_value,
            ],
            dim=time_dim,
        )
        signal_padded = torch.cat([signal_matrix, signal_pad], dim=time_dim)
        subsignal_mask = torch.tril(torch.ones([T + interval[1] + 1, T], device=device))
        time_interval_mask = torch.triu(
            torch.ones([T + interval[1] + 1, T], device=device), -interval[-1] - offset
        ) * torch.tril(
            torch.ones([T + interval[1] + 1, T], device=device), -interval[0]
        )
        masked_signal_matrix = torch.where(
            (time_interval_mask * subsignal_mask) == 1.0, signal_padded, mask_value
        )
        return self.semantics.conjunction(masked_signal_matrix, dim=time_dim, keepdim=False)

    def _next_function(self) -> list[object]:
        return [self.subformula]

    def __str__(self) -> str:
        return "\u25fb " + str(self._interval) + "( " + str(self.subformula) + " )"


class Until(STLFormula):
    """Until: ``phi1`` holds until ``phi2`` becomes true within the interval."""

    def __init__(
        self,
        subformula1: Optional[STLFormula] = None,
        subformula2: Optional[STLFormula] = None,
        interval: Optional[list[Union[int, float]]] = None,
        semantics: Optional[Semantics] = None,
    ) -> None:
        super().__init__(semantics=semantics)
        self.subformula1 = subformula1 or Identity(name="x")
        self.subformula2 = subformula2 or Identity(name="y")
        self.interval = interval
        self._interval = [0, torch.inf] if self.interval is None else self.interval

    def robustness_trace(
        self,
        signal: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        padding: Optional[str] = None,
        large_number: float = 1e9,
        **kwargs: object,
    ) -> torch.Tensor:
        device = signal.device if isinstance(signal, torch.Tensor) else signal[0].device
        time_dim = 0
        conj = self.semantics.conjunction
        disj = self.semantics.disjunction
        conj_id = self.semantics.conjunction_identity

        if isinstance(signal, tuple):
            signal1, signal2 = signal
            assert signal1.shape[time_dim] == signal2.shape[time_dim]
            signal1 = self.subformula1(signal1, padding=padding, large_number=large_number, **kwargs)
            signal2 = self.subformula2(signal2, padding=padding, large_number=large_number, **kwargs)
            T = signal1.shape[time_dim]
        else:
            signal1 = self.subformula1(signal, padding=padding, large_number=large_number, **kwargs)
            signal2 = self.subformula2(signal, padding=padding, large_number=large_number, **kwargs)
            T = signal.shape[time_dim]

        mask_value = conj_id
        if self.interval is None:
            interval = [0, T - 1]
        elif self.interval[1] == torch.inf:
            interval = [self.interval[0], T - 1]
        else:
            interval = self.interval

        # Memory-efficient expand (views, not copies)
        signal1_matrix = signal1.unsqueeze(1).expand(-1, T)
        signal2_matrix = signal2.unsqueeze(1).expand(-1, T)

        if padding == "last":
            pad_value1 = signal1[-1]
            pad_value2 = signal2[-1]
        elif padding == "mean":
            pad_value1 = signal1.mean(dim=time_dim)
            pad_value2 = signal2.mean(dim=time_dim)
        else:
            pad_value1 = torch.tensor(-mask_value, device=device)
            pad_value2 = torch.tensor(-mask_value, device=device)

        signal1_pad = pad_value1.view(1, 1).expand(interval[1] + 1, T)
        signal2_pad = pad_value2.view(1, 1).expand(interval[1] + 1, T)
        signal1_padded = torch.cat([signal1_matrix, signal1_pad], dim=time_dim)
        signal2_padded = torch.cat([signal2_matrix, signal2_pad], dim=time_dim)

        rows = torch.arange(T + interval[1] + 1, device=device).view(-1, 1)
        cols = torch.arange(T, device=device).view(1, -1)

        # Generate masks for each candidate switch point
        phi1_mask = torch.stack(
            [
                (cols - rows >= -end_idx) & (cols - rows <= 0)
                for end_idx in range(interval[0], interval[-1] + 1)
            ],
            dim=0,
        )

        phi2_mask = torch.stack(
            [
                (cols - rows >= -end_idx) & (cols - rows <= -end_idx)
                for end_idx in range(interval[0], interval[-1] + 1)
            ],
            dim=0,
        )

        signal1_batched = signal1_padded.unsqueeze(0)
        signal2_batched = signal2_padded.unsqueeze(0)

        phi1_masked_signal = torch.where(phi1_mask, signal1_batched, mask_value)
        phi2_masked_signal = torch.where(phi2_mask, signal2_batched, mask_value)

        return disj(
            torch.stack(
                [
                    conj(
                        torch.stack(
                            [
                                conj(s1, dim=0, keepdim=False),
                                conj(s2, dim=0, keepdim=False),
                            ],
                            dim=0,
                        ),
                        dim=0,
                        keepdim=False,
                    )
                    for (s1, s2) in zip(phi1_masked_signal, phi2_masked_signal)
                ],
                dim=0,
            ),
            dim=0,
            keepdim=False,
        )

    def _next_function(self) -> list[object]:
        return [self.subformula1, self.subformula2]

    def __str__(self) -> str:
        return (
            "("
            + str(self.subformula1)
            + ") U "
            + str(self._interval)
            + "("
            + str(self.subformula2)
            + ")"
        )


# ==========================================================================
# Recurrent temporal operators (scan-based, O(T) memory)
# ==========================================================================


class TemporalOperator(STLFormula):
    """Base class for recurrent (scan-based) temporal operators."""

    def __init__(
        self,
        subformula: Optional[STLFormula] = None,
        interval: Optional[list[Union[int, float]]] = None,
        semantics: Optional[Semantics] = None,
    ) -> None:
        super().__init__(semantics=semantics)
        self.subformula = subformula or Identity()
        self.interval = interval
        self.interval_str = interval

        if self.interval is None:
            self.hidden_dim: Optional[int] = None
            self._interval: Optional[list[Union[int, float]]] = None
            self.interval_str = [0, torch.inf]
        elif interval[1] == torch.inf:  # type: ignore[index]
            self.hidden_dim = None
            self._interval = [interval[0], interval[1]]  # type: ignore[index]
        else:
            self.hidden_dim = int(interval[1]) + 1  # type: ignore[arg-type, index]
            self._interval = [interval[0], interval[1]]  # type: ignore[index]

        self.LARGE_NUMBER = 1e9
        # Subclasses set this to semantics.conjunction or semantics.disjunction
        self._operation: Optional[Callable[..., torch.Tensor]] = None

    def _get_interval_indices(self) -> tuple[int, Optional[int]]:
        start_idx = -self.hidden_dim  # type: ignore[operator]
        end_idx = -self._interval[0]  # type: ignore[index]
        return start_idx, (None if end_idx == 0 else end_idx)

    def _run_cell(self, signal: torch.Tensor, padding: Optional[str] = None, **kwargs: object) -> torch.Tensor:
        hidden_state = self._initialize_hidden_state(signal, padding=padding)

        def f_(hidden: object, state: object) -> tuple[object, object]:
            hidden, o = self._cell(state, hidden, **kwargs)
            return hidden, o

        _, outputs_stack = scan(f_, hidden_state, signal)
        return outputs_stack

    def _initialize_hidden_state(self, signal: torch.Tensor, padding: Optional[str] = None) -> torch.Tensor:
        device = signal.device

        if padding == "last":
            pad_value = signal[0].detach()
        elif padding == "mean":
            pad_value = signal.mean(0).detach()
        else:
            pad_value = -self.LARGE_NUMBER

        n_time_steps = signal.shape[0]

        if (self.interval is None) or (self.interval[1] == torch.inf):
            self.hidden_dim = n_time_steps
        if self.interval is None:
            self._interval = [0, n_time_steps - 1]
        elif self.interval[1] == torch.inf:
            self._interval[1] = n_time_steps - 1  # type: ignore[index]

        self.M = torch.diag(torch.ones(self.hidden_dim - 1, device=device), diagonal=1)  # type: ignore[operator]
        self.b = torch.zeros(self.hidden_dim, device=device)  # type: ignore[arg-type]
        self.b[-1] = 1.0

        if (self.interval is None) or (self.interval[1] == torch.inf):
            pad_value = torch.cat(
                [
                    torch.ones(self._interval[0] + 1, device=device) * pad_value,  # type: ignore[index]
                    torch.ones(self.hidden_dim - self._interval[0] - 1, device=device)  # type: ignore[operator, index]
                    * self._sign
                    * pad_value,
                ]
            )

        h0 = torch.ones(self.hidden_dim, device=device) * pad_value  # type: ignore[arg-type]
        return h0

    def _cell(self, state: torch.Tensor, hidden: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._operation is not None
        h_new = self.M @ hidden + self.b * state
        start_idx, end_idx = self._get_interval_indices()
        output = self._operation(h_new[start_idx:end_idx], dim=0, keepdim=False)
        return h_new, output

    def robustness_trace(self, signal: torch.Tensor, padding: Optional[str] = None, **kwargs: object) -> torch.Tensor:
        trace = self.subformula(signal, **kwargs)
        outputs = self._run_cell(trace, padding, **kwargs)
        return outputs

    def robustness(self, signal: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.__call__(signal, **kwargs)[-1]

    def _next_function(self) -> list[object]:
        return [self.subformula]


class AlwaysRecurrent(TemporalOperator):
    """Recurrent (scan-based) Always operator."""

    def __init__(
        self,
        subformula: Optional[STLFormula] = None,
        interval: Optional[list[Union[int, float]]] = None,
        semantics: Optional[Semantics] = None,
    ) -> None:
        super().__init__(subformula=subformula, interval=interval, semantics=semantics)
        self._operation = self.semantics.conjunction
        self._sign = -1.0

    def __str__(self) -> str:
        return "\u25fb " + str(self.interval_str) + "( " + str(self.subformula) + " )"


class EventuallyRecurrent(TemporalOperator):
    """Recurrent (scan-based) Eventually operator."""

    def __init__(
        self,
        subformula: Optional[STLFormula] = None,
        interval: Optional[list[Union[int, float]]] = None,
        semantics: Optional[Semantics] = None,
    ) -> None:
        super().__init__(subformula=subformula, interval=interval, semantics=semantics)
        self._operation = self.semantics.disjunction
        self._sign = 1.0

    def __str__(self) -> str:
        return "\u2662 " + str(self.interval_str) + "( " + str(self.subformula) + " )"


class UntilRecurrent(STLFormula):
    """Recurrent (scan-based) Until operator."""

    def __init__(
        self,
        subformula1: Optional[STLFormula] = None,
        subformula2: Optional[STLFormula] = None,
        interval: Optional[list[Union[int, float]]] = None,
        overlap: bool = True,
        semantics: Optional[Semantics] = None,
    ) -> None:
        super().__init__(semantics=semantics)
        self.subformula1 = subformula1 or Identity(name="x")
        self.subformula2 = subformula2 or Identity(name="y")
        self.interval = interval
        if not overlap:
            self.subformula2 = Eventually(
                subformula=self.subformula2, interval=[0, 1], semantics=semantics
            )
        self.LARGE_NUMBER = 1e9
        self.Alw = AlwaysRecurrent(
            GreaterThan(Predicate("x", lambda x: x), 0.0),
            semantics=semantics,
        )

        if self.interval is None:
            self.hidden_dim: Optional[int] = None
        elif interval[1] == torch.inf:  # type: ignore[index]
            self.hidden_dim = None
        else:
            self.hidden_dim = int(interval[1]) + 1  # type: ignore[arg-type, index]

    def _initialize_hidden_state(
        self, signal: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]], padding: Optional[str] = None, **kwargs: object
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        time_dim = 0

        if isinstance(signal, tuple):
            assert signal[0].shape[time_dim] == signal[1].shape[time_dim]
            trace1 = self.subformula1(signal[0], **kwargs)
            trace2 = self.subformula2(signal[1], **kwargs)
            n_time_steps = signal[0].shape[time_dim]
            device = signal[0].device
        else:
            trace1 = self.subformula1(signal, **kwargs)
            trace2 = self.subformula2(signal, **kwargs)
            n_time_steps = signal.shape[time_dim]
            device = signal.device

        if self.hidden_dim is None:
            self.hidden_dim = n_time_steps
        if self.interval is None:
            self.interval = [0, n_time_steps - 1]
        elif self.interval[1] == torch.inf:
            self.interval[1] = n_time_steps - 1

        self.ones_array = torch.ones(self.hidden_dim, device=device)
        self.M = torch.diag(torch.ones(self.hidden_dim - 1, device=device), diagonal=1)
        self.b = torch.zeros(self.hidden_dim, device=device)
        self.b[-1] = 1.0

        if self.hidden_dim == n_time_steps:
            pad_value = self.LARGE_NUMBER
        else:
            pad_value = -self.LARGE_NUMBER

        h1 = pad_value * self.ones_array
        h2 = -self.LARGE_NUMBER * self.ones_array
        return (h1, h2), trace1, trace2

    def _get_interval_indices(self) -> tuple[int, Optional[int]]:
        start_idx = -self.hidden_dim  # type: ignore[operator]
        end_idx = -self.interval[0]  # type: ignore[index]
        return start_idx, (None if end_idx == 0 else end_idx)

    def _cell(
        self, state: tuple[torch.Tensor, torch.Tensor], hidden: tuple[torch.Tensor, torch.Tensor], **kwargs: object
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        conj = self.semantics.conjunction
        disj = self.semantics.disjunction

        x1, x2 = state
        h1, h2 = hidden
        h1_new = self.M @ h1 + self.b * x1
        h1_min = self.Alw(h1_new.flip(0), **kwargs).flip(0)
        h2_new = self.M @ h2 + self.b * x2
        start_idx, end_idx = self._get_interval_indices()
        z = conj(torch.stack([h1_min, h2_new]), dim=0, keepdim=False)[start_idx:end_idx]
        output = disj(z, dim=0, keepdim=False)
        return output, (h1_new, h2_new)

    def robustness_trace(
        self, signal: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]], padding: Optional[str] = None, **kwargs: object
    ) -> torch.Tensor:
        hidden_state, trace1, trace2 = self._initialize_hidden_state(signal, padding=padding, **kwargs)

        def f_(hidden: object, state: object) -> tuple[object, object]:
            o, hidden = self._cell(state, hidden, **kwargs)  # type: ignore[arg-type]
            return hidden, o

        _, outputs_stack = scan(f_, hidden_state, torch.stack([trace1, trace2], dim=1))
        return outputs_stack

    def robustness(self, signal: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.__call__(signal, **kwargs)[-1]

    def _next_function(self) -> list[object]:
        return [self.subformula1, self.subformula2]

    def __str__(self) -> str:
        return "(" + str(self.subformula1) + ") U (" + str(self.subformula2) + ")"


# ==========================================================================
# Differentiable temporal operators (soft time boundaries)
# ==========================================================================


class DifferentiableAlways(STLFormula):
    """Always with smooth (sigmoid-based) time boundaries.

    Instead of crisp integer interval bounds, accepts continuous ``t_start``
    and ``t_end`` parameters, enabling gradient-based optimisation of the
    temporal window itself.
    """

    def __init__(
        self,
        subformula: Optional[STLFormula] = None,
        interval: Optional[list[Union[int, float]]] = None,
        semantics: Optional[Semantics] = None,
    ) -> None:
        super().__init__(semantics=semantics)
        self.interval = interval
        self.subformula = subformula or Identity()

    def robustness_trace(
        self,
        signal: torch.Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
        scale: float = 1.0,
        padding: Optional[str] = None,
        large_number: float = 1e9,
        delta: float = 1e-3,
        **kwargs: object,
    ) -> torch.Tensor:
        device = signal.device
        time_dim = 0
        signal = self.subformula(signal, padding=padding, large_number=large_number, **kwargs)
        T = signal.shape[time_dim]
        mask_value = self.semantics.conjunction_identity
        signal_matrix = signal.reshape([T, 1]) @ torch.ones([1, T], device=device)

        if padding == "last":
            pad_value = signal[-1]
        elif padding == "mean":
            pad_value = signal.mean(time_dim)
        else:
            pad_value = -mask_value

        signal_pad = torch.ones([T, T], device=device) * pad_value
        signal_padded = torch.cat([signal_matrix, signal_pad], dim=time_dim)
        smooth_time_mask = smooth_mask(T, t_start, t_end, scale, device=device)
        padded_smooth_time_mask = torch.stack(
            [
                torch.concat(
                    [
                        torch.zeros(i, device=device),
                        smooth_time_mask,
                        torch.zeros(T - i, device=device),
                    ]
                )
                for i in range(T)
            ],
            1,
        )
        masked_signal_matrix = torch.where(
            padded_smooth_time_mask > delta,
            signal_padded * padded_smooth_time_mask,
            mask_value,
        )
        return self.semantics.conjunction(masked_signal_matrix, dim=time_dim, keepdim=False)

    def _next_function(self) -> list[object]:
        return [self.subformula]

    def __str__(self) -> str:
        return "\u25fb [a,b] ( " + str(self.subformula) + " )"


class DifferentiableEventually(STLFormula):
    """Eventually with smooth (sigmoid-based) time boundaries."""

    def __init__(
        self,
        subformula: Optional[STLFormula] = None,
        interval: Optional[list[Union[int, float]]] = None,
        semantics: Optional[Semantics] = None,
    ) -> None:
        super().__init__(semantics=semantics)
        self.interval = interval
        self.subformula = subformula or Identity()
        self._interval = [0, torch.inf] if self.interval is None else self.interval

    def robustness_trace(
        self,
        signal: torch.Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
        scale: float = 1.0,
        padding: Optional[str] = None,
        large_number: float = 1e9,
        delta: float = 1e-3,
        **kwargs: object,
    ) -> torch.Tensor:
        device = signal.device
        time_dim = 0
        signal = self.subformula(signal, padding=padding, large_number=large_number, **kwargs)
        T = signal.shape[time_dim]
        mask_value = self.semantics.disjunction_identity
        signal_matrix = signal.reshape([T, 1]) @ torch.ones([1, T], device=device)

        if padding == "last":
            pad_value = signal[-1]
        elif padding == "mean":
            pad_value = signal.mean(time_dim)
        else:
            pad_value = mask_value

        signal_pad = torch.ones([T, T], device=device) * pad_value
        signal_padded = torch.cat([signal_matrix, signal_pad], dim=time_dim)
        smooth_time_mask = smooth_mask(T, t_start, t_end, scale, device=device)
        padded_smooth_time_mask = torch.stack(
            [
                torch.concat(
                    [
                        torch.zeros(i, device=device),
                        smooth_time_mask,
                        torch.zeros(T - i, device=device),
                    ]
                )
                for i in range(T)
            ],
            1,
        )
        masked_signal_matrix = torch.where(
            padded_smooth_time_mask > delta,
            signal_padded * padded_smooth_time_mask,
            mask_value,
        )
        return self.semantics.disjunction(masked_signal_matrix, dim=time_dim, keepdim=False)

    def _next_function(self) -> list[object]:
        return [self.subformula]

    def __str__(self) -> str:
        return "\u2662 [a,b] ( " + str(self.subformula) + " )"
