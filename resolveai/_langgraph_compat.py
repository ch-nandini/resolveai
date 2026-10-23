"""Minimal stand-in for langgraph's StateGraph, used ONLY if `langgraph` isn't installed.

Supports the subset this project uses: TypedDict state with Annotated reducers, add_node,
add_edge (incl. START/END), add_conditional_edges, compile(checkpointer=...), invoke(config=...).
Install the real thing with `pip install langgraph` and this file is never imported.
"""
import typing

START = "__start__"
END = "__end__"


class GraphRecursionError(RuntimeError):
    pass


class MemorySaver:  # no-op placeholder for the real checkpointer
    pass


class StateGraph:
    def __init__(self, state_schema):
        self.reducers = {}
        hints = typing.get_type_hints(state_schema, include_extras=True)
        for key, hint in hints.items():
            meta = getattr(hint, "__metadata__", None)
            if meta and callable(meta[0]):
                self.reducers[key] = meta[0]
        self.nodes, self.edges, self.cond = {}, {}, {}

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, src, dst):
        self.edges[src] = dst

    def add_conditional_edges(self, src, fn, mapping=None):
        self.cond[src] = (fn, mapping)

    def compile(self, checkpointer=None):
        return _Compiled(self)


class _Compiled:
    def __init__(self, g):
        self.g = g

    def _merge(self, state, update):
        for k, v in update.items():
            if k in self.g.reducers and k in state:
                state[k] = self.g.reducers[k](state[k], v)
            else:
                state[k] = v
        return state

    def _next(self, node, state):
        if node in self.g.cond:
            fn, mapping = self.g.cond[node]
            out = fn(state)
            return mapping[out] if mapping else out
        return self.g.edges.get(node, END)

    def invoke(self, state, config=None):
        limit = (config or {}).get("recursion_limit", 25)
        state = dict(state)
        node, n = self.g.edges[START], 0
        while node != END:
            n += 1
            if n > limit:
                raise GraphRecursionError(f"Recursion limit of {limit} reached")
            state = self._merge(state, self.g.nodes[node](state) or {})
            node = self._next(node, state)
        return state
