"""Counterfactual label generation.

This module is part of the label identity (see pipeline.simulator_identity):
changing it changes what a label means. Orchestration code is not.

A label for (state at t, candidate, hold) is the loss added after t when the
candidate controls the fleet until the holding boundary, the continuation
policy controls it afterwards, only orders released through t + H are
admitted, and every admitted order is finally delivered.
"""
from __future__ import annotations

import math
import time

LABEL_VERSION = 2


def prepare_tail(runtime, before, *, candidate, boundary, horizon, continuation, keep_boundary):
    """Run one candidate from the shared snapshot to its holding boundary.

    Returns the boundary snapshot (taken before the arrival cutoff, only when
    requested), the digest of the scoring-tail start state and the window wall
    time. The runtime is left at the start of the scoring tail.
    """
    runtime.restore(before)
    runtime.set_policy(candidate)
    started = time.monotonic()
    runtime.advance(boundary)
    state = runtime.snapshot() if keep_boundary else None
    runtime.cutoff_arrivals(horizon)
    runtime.set_policy(continuation)
    return state, runtime.state_digest(), time.monotonic() - started


def score_tail(runtime, *, horizon, start_loss):
    """Finish the prepared tail; a failed or incomplete drain is an error."""
    started = time.monotonic()
    runtime.advance(max(horizon, runtime.now))
    loss = runtime.drain() - start_loss
    if not math.isfinite(loss) or loss < -1e-8:
        raise ValueError("invalid incremental rollout loss")
    return max(0., loss), time.monotonic() - started
