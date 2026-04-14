# vehicle-stl

Signal Temporal Logic (STL) evaluation with pluggable semantics for PyTorch.

Forked from [stlcg++](https://github.com/UW-CTRL/stlcg-plus-plus) by Karen Leung et al.

## Key changes from stlcg++

- **Pluggable semantics**: Conjunction/disjunction reductions are parameterised via a `Semantics` object. Choose between exact min/max, softmax, logsumexp, or provide custom reduction functions.
- **Clean API**: All formula classes accept an optional `semantics` argument. Subformulas default to `Identity()` for direct robustness-tensor input.
- **Code quality**: Explicit imports, type annotations, fixed bugs from upstream.
- **Full feature preservation**: Expression, Predicate, recurrent variants, differentiable variants, and visualization are all retained.

## Usage

```python
import torch
from vehicle_stl import Always, Eventually, Until, Predicate, EXACT, softmax

# Exact semantics (default)
signal = torch.tensor([0.5, 0.3, -0.2, 0.4, 0.1])
formula = Always(interval=[0, 2])
result = formula(signal)

# Smooth semantics for gradient-based optimisation
smooth = softmax(temperature=10.0)
formula = Always(interval=[0, 2], semantics=smooth)
result = formula(signal)

# With predicates (stlcg++ style)
pred = Predicate("x", lambda s: s)
formula = Always(pred > 0.0, interval=[0, 2], semantics=smooth)
result = formula(signal)
```

## License

MIT (same as upstream stlcg++).
