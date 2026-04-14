"""vehicle-stl: Signal Temporal Logic evaluation with pluggable semantics.

Forked from stlcg++ (https://github.com/UW-CTRL/stlcg-plus-plus).
"""

from .formula import (
    Always,
    AlwaysRecurrent,
    And,
    DifferentiableAlways,
    DifferentiableEventually,
    Equal,
    Eventually,
    EventuallyRecurrent,
    Expression,
    GreaterThan,
    Identity,
    Implies,
    LessThan,
    Negation,
    Or,
    Predicate,
    STLFormula,
    TemporalOperator,
    Until,
    UntilRecurrent,
)
from .semantics import (
    EXACT,
    Semantics,
    logsumexp,
    softmax,
)

__all__ = [
    # Semantics
    "EXACT",
    "Semantics",
    "logsumexp",
    "softmax",
    # Formula base
    "STLFormula",
    "Identity",
    "Expression",
    "Predicate",
    # Comparators
    "LessThan",
    "GreaterThan",
    "Equal",
    # Propositional
    "Negation",
    "And",
    "Or",
    "Implies",
    # Standard temporal (matrix-based)
    "Always",
    "Eventually",
    "Until",
    # Recurrent temporal (scan-based)
    "TemporalOperator",
    "AlwaysRecurrent",
    "EventuallyRecurrent",
    "UntilRecurrent",
    # Differentiable temporal (soft boundaries)
    "DifferentiableAlways",
    "DifferentiableEventually",
]
