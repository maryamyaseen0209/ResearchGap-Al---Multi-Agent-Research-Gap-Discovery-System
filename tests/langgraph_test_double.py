"""TEST-ONLY miniature execution double for offline routing checks.

This is not LangGraph and must never be shipped/imported by production code.
Tests prove our nodes/routes with a stand-in; real LangGraph execution must be
verified in an environment where the dependency can be installed.
"""

from copy import deepcopy
from types import ModuleType
import sys


class InMemorySaver:
    pass


class FakeCompiledGraph:
    def __init__(self, builder, checkpointer):
        self.builder = builder
        self.checkpointer = checkpointer
        self.states = {}

    def invoke(self, payload, config):
        thread_id = config["configurable"]["thread_id"]
        if payload is None:
            return deepcopy(self.states.get(thread_id))
        current = deepcopy(payload)
        next_node = self.builder.start
        visited = 0
        while next_node != "__end__":
            visited += 1
            if visited > 15:
                raise RuntimeError("Fake graph detected unexpected loop")
            patch = self.builder.nodes[next_node](current)
            for key, value in patch.items():
                if key == "events":
                    current.setdefault("events", []).extend(value)
                else:
                    current[key] = deepcopy(value)
            if next_node in self.builder.conditionals:
                func, mapping = self.builder.conditionals[next_node]
                next_node = mapping[func(current)]
            elif next_node in self.builder.edges:
                next_node = self.builder.edges[next_node]
            else:
                raise RuntimeError("Fake graph missing outgoing edge")
        self.states[thread_id] = deepcopy(current)
        return deepcopy(current)


class FakeStateGraph:
    def __init__(self, schema):
        self.schema = schema
        self.nodes = {}
        self.edges = {}
        self.conditionals = {}
        self.start = None

    def add_node(self, name, fn):
        self.nodes[name] = fn
        return self

    def add_edge(self, src, dest):
        if src == "__start__":
            self.start = dest
        else:
            self.edges[src] = dest
        return self

    def add_conditional_edges(self, src, fn, mapping):
        self.conditionals[src] = (fn, mapping)
        return self

    def compile(self, checkpointer=None):
        return FakeCompiledGraph(self, checkpointer)


def install(monkeypatch):
    pkg = ModuleType("langgraph")
    pkg.__path__ = []
    graph = ModuleType("langgraph.graph")
    graph.START = "__start__"
    graph.END = "__end__"
    graph.StateGraph = FakeStateGraph
    checkpoint = ModuleType("langgraph.checkpoint")
    checkpoint.__path__ = []
    memory = ModuleType("langgraph.checkpoint.memory")
    memory.InMemorySaver = InMemorySaver
    monkeypatch.setitem(sys.modules, "langgraph", pkg)
    monkeypatch.setitem(sys.modules, "langgraph.graph", graph)
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint", checkpoint)
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.memory", memory)
