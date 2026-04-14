"""Visualization of STL computation graphs.

Requires the ``graphviz`` package (install with ``pip install vehicle-stl[viz]``).
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any, Optional

import torch

from .formula import Expression, STLFormula

Node = namedtuple("Node", ("name", "inputs", "attr", "op"))


class Legend:
    def __init__(self, name: str, color: str, next: Optional[Legend] = None) -> None:
        self.name = name
        self.color = color
        self.next = next

    def _next_legend(self) -> list[Legend]:
        if self.next is None:
            return []
        return [self.next]


def make_stl_graph(
    form: Any,
    node_attr: Optional[dict[str, str]] = None,
    graph_attr: Optional[dict[str, str]] = None,
    show_legend: bool = False,
) -> Any:
    """Produce a Graphviz ``Digraph`` of the STL formula tree.

    Returns a ``graphviz.Digraph`` object that can be rendered with
    :func:`save_graph`.
    """
    from graphviz import Digraph

    if node_attr is None:
        node_attr = dict(
            style="filled",
            shape="box",
            align="left",
            fontsize="12",
            ranksep="0.1",
            height="0.2",
            fontname="monospace",
        )
    if graph_attr is None:
        graph_attr = dict(size="12,12")

    dot = Digraph(node_attr=node_attr, graph_attr=graph_attr)

    def add_nodes(form: Any) -> None:
        if isinstance(form, torch.Tensor):
            dot.node(str(id(form)), str(form), fillcolor="lightskyblue")
        elif isinstance(form, Expression):
            dot.node(str(id(form)), form.name, fillcolor="lightskyblue")
        elif isinstance(form, str):
            dot.node(str(id(form)), form, fillcolor="lightskyblue")
        elif isinstance(form, STLFormula):
            dot.node(
                str(id(form)),
                form.__class__.__name__ + "\n" + str(form),
                fillcolor="orange",
            )
        elif isinstance(form, Legend):
            dot.node(str(id(form)), form.name, fillcolor=form.color, color="white")
        else:
            dot.node(str(id(form)), str(form), fillcolor="palegreen")

        if hasattr(form, "_next_function"):
            for u in form._next_function():
                dot.edge(str(id(u)), str(id(form)))
                add_nodes(u)

        if hasattr(form, "_next_legend"):
            for u in form._next_legend():
                dot.edge(str(id(u)), str(id(form)), color="white")
                add_nodes(u)

    legend_names = ["expression", "constant", "formula"]
    legend_colors = ["lightskyblue", "palegreen", "orange"]
    legends = [Legend(legend_names[0], legend_colors[0])]
    for i in range(1, 3):
        legends.append(Legend(legend_names[i], legend_colors[i], legends[i - 1]))

    if show_legend:
        form = (form, legends[-1])
    if isinstance(form, tuple):
        for v in form:
            add_nodes(v)
    else:
        add_nodes(form)
    resize_graph(dot)

    return dot


def resize_graph(dot: Any, size_per_element: float = 0.15, min_size: float = 12) -> None:
    """Resize graph according to content."""
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    dot.graph_attr.update(size=f"{size},{size}")


def save_graph(dot: Any, filename: str, format: str = "pdf", cleanup: bool = True) -> None:
    """Render and save an STL computation graph."""
    dot.render(filename=filename, format=format, cleanup=cleanup)
