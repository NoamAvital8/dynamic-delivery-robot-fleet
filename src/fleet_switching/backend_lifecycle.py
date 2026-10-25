"""Idempotent hooks and owned-cache cleanup without editing the pinned backend."""
from __future__ import annotations

import functools


def install(fast):
    if getattr(fast, "_switching_lifecycle_installed", False):
        return
    graph_install = fast._install_in_memory_graph_initialization
    distance_install = fast._install_compiled_distance_backend
    state = {"graph": False, "distance": None}

    @functools.wraps(graph_install)
    def graph_once():
        if not state["graph"]:
            graph_install()
            state["graph"] = True

    @functools.wraps(distance_install)
    def distance_once():
        if state["distance"] is None:
            state["distance"] = distance_install()
        return state["distance"]

    fast._install_in_memory_graph_initialization = graph_once
    fast._install_compiled_distance_backend = distance_once
    fast._switching_lifecycle_installed = True


def release(fast, graph, oracle):
    registry = fast._oracle_by_graph_id
    if registry.get(id(graph)) is oracle:
        del registry[id(graph)]
    for key in list(fast._distance_rows):
        if key[0] == id(oracle):
            del fast._distance_rows[key]
