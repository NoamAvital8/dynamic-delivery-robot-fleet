"""Exact memoization for repeated charger-distance queries.

Reactive insertion evaluates many hypothetical routes over the same service
nodes.  ``ChargerDistanceIndex.distances_to_stations`` is numerically cheap per
call when the dense index is available, but converting the dense NumPy row into
a filtered Python dict hundreds of thousands of times is not cheap.  The
results are immutable for a fixed road graph, source node and cutoff, so they
can be memoized without changing routing semantics.
"""

from __future__ import annotations

from functools import lru_cache

from .routing import ChargerDistanceIndex

_INSTALLED = False
_ORIGINAL = None


def install_charger_distance_query_cache(maxsize: int = 100_000) -> None:
    """Memoize ``distances_to_stations`` exactly for the current process.

    The method's returned dictionaries are treated as read-only throughout the
    simulator.  Cache keys include the index object, source, cutoff and
    ``include_self`` flag, so no approximation or cutoff rounding is used.
    """
    global _INSTALLED, _ORIGINAL
    if _INSTALLED:
        return

    original = ChargerDistanceIndex.distances_to_stations
    _ORIGINAL = original

    @lru_cache(maxsize=maxsize)
    def cached(self, source, cutoff_m=None, include_self=True):
        return original(
            self,
            source,
            cutoff_m=cutoff_m,
            include_self=include_self,
        )

    ChargerDistanceIndex.distances_to_stations = cached
    _INSTALLED = True
