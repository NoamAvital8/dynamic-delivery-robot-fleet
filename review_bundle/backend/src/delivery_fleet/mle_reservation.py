"""Anytime Poisson rates and a bounded, uncertainty-gated alpha search.

Only the Poisson confidence sets have a coverage guarantee. Finite stress
scenarios and empirical queue rollouts are an approximate decision model, not
a confidence certificate for simulator loss or a global optimizer.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import itertools
import math
import multiprocessing
from typing import Hashable, Sequence

import numpy as np


@dataclass(frozen=True)
class RateBounds:
    count: int
    exposure_min: float
    mle: float
    lower: float
    upper: float


def poisson_confidence_sequence(count: int, exposure_min: float, error: float,
                                *, shape: float = .5, rate_min: float = 10.) -> RateBounds:
    """Invert a fixed Gamma-mixture likelihood-ratio martingale.

    log M(lambda) = a log b + lgamma(a+n)-lgamma(a)
                    -(a+n) log(b+t) + lambda*t - n*log(lambda).
    Ville's inequality bounds crossing 1/error at ANY inspection time under
    a constant-rate Poisson process. The mixing distribution is fixed before
    observing data; it is not a fitted posterior or NN.
    """
    if isinstance(count, bool) or int(count) != count or count < 0:
        raise ValueError("count must be a non-negative integer")
    if (not math.isfinite(exposure_min) or exposure_min < 0 or
            not 0 < error < 1 or not math.isfinite(shape) or shape <= 0 or
            not math.isfinite(rate_min) or rate_min <= 0):
        raise ValueError("invalid exposure, error probability or mixture parameters")
    n, t = int(count), float(exposure_min)
    if t == 0:
        if n:
            raise ValueError("positive counts require positive exposure")
        return RateBounds(0, 0., 0., 0., math.inf)
    constant = (shape*math.log(rate_min) + math.lgamma(shape+n)
                - math.lgamma(shape) - (shape+n)*math.log(rate_min+t))
    boundary = -math.log(error)
    mle = n/t

    def objective(lam):
        if lam == 0:
            return constant-boundary if n == 0 else math.inf
        return constant+lam*t-n*math.log(lam)-boundary

    if n == 0:
        return RateBounds(n, t, mle, 0., (boundary-constant)/t)
    lo, hi = 0., mle
    for _ in range(80):
        mid = (lo+hi)/2
        if objective(mid) > 0:
            lo = mid
        else:
            hi = mid
    lower = hi
    lo, hi = mle, max(2*mle, 1/t)
    while objective(hi) <= 0:
        hi *= 2
    for _ in range(80):
        mid = (lo+hi)/2
        if objective(mid) > 0:
            hi = mid
        else:
            lo = mid
    return RateBounds(n, t, mle, lower, hi)


class EpochPoissonRates:
    """Fixed calendar epochs with a joint error budget over cells and time.

    Cell error in epoch e is delta * 6/(pi^2*(e+1)^2) / family_size.
    No adaptive window selection. Constant Poisson rates within each fixed
    epoch are an assumption, not a consequence of narrow intervals.
    """
    def __init__(self, cells: Sequence[Hashable], *, confidence: float = .95,
                 epoch_min: float = 60., family_size: int | None = None):
        self.cells = tuple(cells)
        if len(set(self.cells)) != len(self.cells) or not self.cells:
            raise ValueError("cells must be nonempty and unique")
        if not math.isfinite(confidence) or not 0 < confidence < 1:
            raise ValueError("confidence must be strictly between zero and one")
        if not math.isfinite(epoch_min) or epoch_min <= 0:
            raise ValueError("epoch length must be finite and positive")
        self.family_size = len(self.cells) if family_size is None else int(family_size)
        if self.family_size < len(self.cells):
            raise ValueError("family size cannot be smaller than the cell count")
        self.confidence, self.epoch_min = confidence, float(epoch_min)
        self.epoch, self.last_time = 0, 0.
        self.counts = dict.fromkeys(self.cells, 0)

    def advance(self, now: float) -> bool:
        if not math.isfinite(now) or now < self.last_time:
            raise ValueError("observation/inspection times must be finite and monotone")
        epoch = int(now // self.epoch_min)
        changed = epoch != self.epoch
        if changed:
            self.counts = dict.fromkeys(self.cells, 0)
            self.epoch = epoch
        self.last_time = now
        return changed

    def observe(self, cell: Hashable, now: float):
        if cell not in self.counts:
            raise ValueError("unknown arrival cell")
        self.advance(float(now))
        self.counts[cell] += 1

    def bounds(self, now: float) -> dict[Hashable, RateBounds]:
        self.advance(float(now))
        t = now-self.epoch*self.epoch_min
        error = (1-self.confidence)*6/(math.pi**2*(self.epoch+1)**2)/self.family_size
        # Events exactly on the epoch boundary have zero exposure. Do not
        # fabricate a rate or attempt to activate reservations at that instant.
        if t == 0:
            return {cell: RateBounds(n, 0., 0., 0., math.inf)
                    for cell, n in self.counts.items()}
        return {cell: poisson_confidence_sequence(n, t, error)
                for cell, n in self.counts.items()}


@dataclass(frozen=True)
class MLEConfig:
    confidence: float = .95
    epoch_min: float = 60.
    update_interval_min: float = 15.
    horizon_min: float = 45.
    minimum_arrivals: int = 20
    maximum_relative_rate_width: float = .75
    stability_tolerance: float = .10
    minimum_gain_fraction: float = .01
    rollouts: int = 4
    max_requests_per_trace: int = 1000
    processes: int = 1
    seed: int = 20261007

    def __post_init__(self):
        if not math.isfinite(self.confidence) or not 0 < self.confidence < 1:
            raise ValueError("confidence must lie strictly between zero and one")
        for name in ['epoch_min', 'update_interval_min', 'horizon_min']:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ['minimum_arrivals', 'rollouts', 'max_requests_per_trace', 'processes']:
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ['stability_tolerance', 'minimum_gain_fraction']:
            v = getattr(self, name)
            if not math.isfinite(v) or not 0 <= v <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if (not math.isfinite(self.maximum_relative_rate_width)
                or self.maximum_relative_rate_width <= 0):
            raise ValueError('maximum_relative_rate_width must be finite and positive')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError('seed must be a non-negative integer')


def add_mle_arguments(parser):
    """Shared flags: direct simulator and multicity controller stay in sync."""
    for name, default in MLEConfig().__dict__.items():
        parser.add_argument('--mle-'+name.replace('_', '-'), type=type(default), default=default)


def mle_config_from_args(args) -> MLEConfig:
    return MLEConfig(**{name: getattr(args, 'mle_'+name) for name in MLEConfig.__dataclass_fields__})


@dataclass(frozen=True)
class SurrogateTrace:
    arrivals: np.ndarray
    importance: np.ndarray
    allowance: np.ndarray
    service: np.ndarray  # request x robot; inf denotes incapable
    ready: np.ndarray   # initial remaining mission/charging availability


def surrogate_loss(trace: SurrogateTrace, thresholds: np.ndarray) -> float:
    """Nonpreemptive server approximation; NOT the graph/insertion simulator.

    Known busy routes are represented by ready times, never canceled here.
    Each synthetic job has fixed per-robot response/service estimates. Future
    movement, insertion, battery depletion and induced charger traffic are not
    propagated between jobs. This deliberate bound keeps online search cheap.
    """
    available = trace.ready.copy()
    loss = 0.
    for i, arrival in enumerate(trace.arrivals):
        eligible = thresholds <= trace.importance[i]
        completion = np.maximum(available, arrival)+trace.service[i]
        elapsed = completion-arrival
        w, deadline = trace.importance[i], trace.allowance[i]
        values = w*np.minimum(elapsed, deadline)+(w+1)**2*np.maximum(0., elapsed-deadline)
        values = np.where(eligible, values, np.inf)
        chosen = int(np.argmin(values))
        if not math.isfinite(float(values[chosen])):
            return math.inf
        available[chosen] = completion[chosen]
        loss += float(values[chosen])
    return loss


def _score_candidate(job):
    thresholds, scenarios = job
    return [float(np.mean([surrogate_loss(trace, thresholds) for trace in traces]))
            for traces in scenarios]


@dataclass(frozen=True)
class GateDecision:
    enabled: bool
    candidate_index: int
    reason: str
    minimum_gain_fraction: float
    maximum_alpha_disagreement: float


def select_gated_candidate(scores: np.ndarray, fractions: np.ndarray,
                           config: MLEConfig) -> GateDecision:
    """Central optimum, alpha stability and improvement on EVERY stress case.

    Candidate zero is the unrestricted greedy fallback. Stress-grid coverage
    is heuristic: passing is not a proof over every point in the confidence box.
    """
    if (scores.ndim != 2 or fractions.ndim < 2 or scores.shape[0] == 0
            or fractions.shape[0] != scores.shape[0] or fractions.size == 0
            or scores.shape[1] == 0 or not np.all(np.isfinite(scores))
            or not np.all(np.isfinite(fractions))):
        return GateDecision(False, 0, 'infeasible_surrogate', 0., math.inf)
    central = int(np.argmin(scores[:, 0]))
    if central == 0:
        return GateDecision(False, 0, 'fallback_best', 0., 0.)
    winners = np.argmin(scores, axis=0)
    disagreement = float(np.max(np.abs(fractions[winners]-fractions[central])))
    gains = (scores[0]-scores[central])/np.maximum(scores[0], 1e-12)
    gain = float(np.min(gains))
    if disagreement > config.stability_tolerance:
        return GateDecision(False, 0, 'unstable_alpha', gain, disagreement)
    if gain <= config.minimum_gain_fraction:
        return GateDecision(False, 0, 'insufficient_stress_gain', gain, disagreement)
    return GateDecision(True, central, 'stable_stress_improvement', gain, disagreement)


def importance_stress_rates(bounds: Sequence[RateBounds]) -> list[np.ndarray]:
    """MLE plus all lower/upper corners of the small importance-level box."""
    result = [np.array([b.mle for b in bounds])]
    result.extend(np.array([b.upper if high else b.lower for b, high in zip(bounds, mask)])
                  for mask in itertools.product([False, True], repeat=len(bounds)))
    return result


class CandidateScorer:
    def __init__(self, processes: int = 1):
        if isinstance(processes, bool) or int(processes) != processes or processes < 1:
            raise ValueError("processes must be a positive integer")
        self.processes = int(processes)
        self._pool = None

    def score(self, thresholds: Sequence[np.ndarray], scenarios) -> np.ndarray:
        jobs = [(t, scenarios) for t in thresholds]
        if self.processes == 1:
            return np.asarray([_score_candidate(job) for job in jobs])
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.processes,
                                            mp_context=multiprocessing.get_context('spawn'))
        return np.asarray(list(self._pool.map(_score_candidate, jobs)))

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
