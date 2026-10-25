"""Loss regression, whole-episode validation and conservative switching.

Default target ("advantage"): for each state, the per-robot loss of every
policy minus the mean over the portfolio. Most of a label (open orders,
future arrivals, the shared continuation) is common to all policies; removing
it lets each head model only the policy-specific part, and switching margins
are calibrated on errors of predicted *differences* between policies.
A separate "level" head predicts the portfolio-mean loss per robot, so
absolute predictions (level + advantage) remain available for reports and the
relative gain threshold. The legacy "absolute" target is kept for ablations.

No softmax probabilities are interpreted as correctness probabilities, and
the validation-residual margins are heuristics, not confidence intervals.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
from pathlib import Path
import pickle

import numpy as np

from .artifacts import atomic_bytes, digest, identity, runtime_versions

TARGETS = ("advantage", "absolute")


def _weighted_quantile(values, weights, q):
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if not len(values):
        return 0.
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, q * cumulative[-1])])


def _pair(a, b):
    return "|".join(sorted((a, b)))


@dataclass
class NumpyRegressor:
    """Two-layer MLP with Adam, sample weights and weighted train-only normalization."""
    seed: int = 42
    hidden: int = 64
    epochs: int = 150
    batch: int = 128
    learning_rate: float = .001

    def fit(self, x, y, sample_weight=None):
        w = np.ones(len(x)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        w = w / w.sum()
        self.mean = (w[:, None] * x).sum(axis=0)
        spread = np.sqrt((w[:, None] * (x - self.mean) ** 2).sum(axis=0))
        self.scale = np.where(spread > 1e-6, spread, 1.)
        self.y_mean = float((w * y).sum())
        self.y_scale = max(float(np.sqrt((w * (y - self.y_mean) ** 2).sum())), 1e-6)
        z, target = (x - self.mean) / self.scale, (y - self.y_mean) / self.y_scale
        rng = np.random.default_rng(self.seed)
        self.parameters = [rng.normal(0, math.sqrt(2 / x.shape[1]), (x.shape[1], self.hidden)),
                           np.zeros(self.hidden), rng.normal(0, math.sqrt(1 / self.hidden), (self.hidden, 1)),
                           np.zeros(1)]
        first = [np.zeros_like(p) for p in self.parameters]
        second = [np.zeros_like(p) for p in self.parameters]
        step = 0
        # Every row is used; episode weights scale each row's squared error.
        relative = w * len(w)
        for _ in range(self.epochs):
            indices = rng.permutation(len(z))
            for start in range(0, len(z), self.batch):
                ii = indices[start:start + self.batch]
                xx, yy, ww = z[ii], target[ii, None], relative[ii, None]
                weights, b, v, c = self.parameters
                hidden = np.maximum(0, xx @ weights + b)
                derivative = 2 * ww * (hidden @ v + c - yy) / max(float(ww.sum()), 1e-12)
                back = (derivative @ v.T) * (hidden > 0)
                gradients = [xx.T @ back, back.sum(axis=0), hidden.T @ derivative, derivative.sum(axis=0)]
                step += 1
                for j, (p, g) in enumerate(zip(self.parameters, gradients)):
                    g = np.clip(g, -10, 10)
                    first[j] = .9 * first[j] + .1 * g
                    second[j] = .999 * second[j] + .001 * g * g
                    p -= self.learning_rate * (first[j] / (1 - .9 ** step)) / (
                        np.sqrt(second[j] / (1 - .999 ** step)) + 1e-8)
        return self

    def predict(self, x):
        w, b, v, c = self.parameters
        return (np.maximum(0, ((x - self.mean) / self.scale) @ w + b) @ v + c).ravel() * self.y_scale + self.y_mean


@dataclass
class LossSelector:
    feature_names: tuple[str, ...]
    policies: tuple[str, ...]
    windows: tuple[float, ...]
    predictors: dict
    validation_error_q90: dict
    default_window: float = 15.
    minimum_gain_fraction: float = .05
    adaptive: bool = False
    provenance: dict | None = None
    target: str = "absolute"
    level_predictor: object = None
    pair_error_q90: dict = field(default_factory=dict)
    margin_scale: float = 1.

    def _matrix(self, features_by_window):
        x = np.array([[features[name] for name in self.feature_names] for features in features_by_window],
                     dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError("selector received non-finite features")
        return x

    def _heads(self, rows):
        x = self._matrix(rows)
        heads = {p: self.predictors[p].predict(x) for p in self.policies}
        if self.target == "advantage":
            level = self.level_predictor.predict(x)
            advantage = [{p: float(heads[p][i]) for p in self.policies} for i in range(len(rows))]
        else:
            level = np.array([np.mean([heads[p][i] for p in self.policies]) for i in range(len(rows))])
            advantage = [{p: float(heads[p][i] - level[i]) for p in self.policies} for i in range(len(rows))]
        return [float(v) for v in level], advantage

    def predictions(self, features):
        """Absolute predicted loss per robot (level + advantage), floored at zero."""
        level, advantage = self._heads([features])
        return {p: max(0., level[0] + advantage[0][p]) for p in self.policies}

    def pair_error(self, a, b):
        if a == b:
            return 0.
        if self.pair_error_q90:
            return self.margin_scale * self.pair_error_q90[_pair(a, b)]
        # Legacy absolute selectors: summed per-policy residual quantiles.
        return self.margin_scale * (self.validation_error_q90[a] + self.validation_error_q90[b])

    def _best(self, advantage):
        return min(self.policies, key=lambda p: (advantage[p], self.policies.index(p)))

    def choose(self, features_by_window, current):
        """Pick (policy, holding minutes). `current=None` means no incumbent yet."""
        if current is not None and current not in self.policies:
            raise ValueError("current policy is outside the trained portfolio")
        windows = sorted(self.windows) if self.adaptive else [self.default_window]
        level, advantage = self._heads([features_by_window[w] for w in windows])
        by_window = {w: (level[i], advantage[i]) for i, w in enumerate(windows)}
        default_level, default_advantage = by_window[self.default_window]
        winner = self._best(default_advantage)
        vetoed = False
        gain = margin = uncertainty = 0.
        if current is not None and winner != current:
            gain = default_advantage[current] - default_advantage[winner]
            margin = self.minimum_gain_fraction * max(default_level + default_advantage[current], 1e-9)
            uncertainty = self.pair_error(current, winner)
            if gain <= max(margin, uncertainty):
                winner, vetoed = current, True
        window = self.default_window
        if self.adaptive:
            # A longer hold needs the same winner, with a margin, at EVERY
            # shorter tested horizon as well (longest consistent prefix).
            eligible = []
            for hold in windows:
                hold_level, hold_advantage = by_window[hold]
                best = self._best(hold_advantage)
                others = [p for p in self.policies if p != best]
                gap = min((hold_advantage[p] - hold_advantage[best] for p in others), default=0.)
                error = max((self.pair_error(best, p) for p in others), default=0.)
                floor = self.minimum_gain_fraction * max(hold_level + hold_advantage[best], 1e-9)
                if best != winner or gap <= max(error, floor):
                    break
                eligible.append(hold)
            spread = max(default_advantage.values()) - min(default_advantage.values())
            if eligible:
                window = max(eligible)
            elif spread > max(uncertainty, margin):
                window = min(self.windows)
            # Nearly equivalent policies keep the default interval, not frantic switching.
        return winner, float(window), {
            "predictions_per_robot": {p: max(0., default_level + default_advantage[p]) for p in self.policies},
            "selected": winner, "holding_min": window, "incumbent": current,
            "switch_vetoed": vetoed, "predicted_gain_per_robot": gain,
            "relative_margin_per_robot": margin, "validation_margin": uncertainty,
            "confidence_interpretation": "validation residual heuristic; no probability guarantee"}

    def save(self, path):
        atomic_bytes(path, pickle.dumps(self, protocol=5))

    @staticmethod
    def load(path):
        with Path(path).open("rb") as stream:
            value = pickle.load(stream)
        if not isinstance(value, LossSelector):
            raise ValueError("not a loss-selector model")
        return value


def _current_policy(row, policies):
    active = [p for p in policies if row["features"].get("state.policy." + p, 0.) == 1.]
    return active[0] if len(active) == 1 else None


def evaluate_selector(selector, records):
    """Offline diagnostics on held-out rows; NOT a substitute for online validation.

    Regret is measured for the raw argmin AND for the deployed `choose()` rule
    (incumbent from the row's observable current-policy features, hysteresis,
    holding rule). Means are reported per decision row and per episode.
    """
    policies = selector.policies
    by_state = defaultdict(dict)
    for row in records:
        by_state[(row["group"], row["step"])][float(row["holding_min"])] = row
    best_fixed = min(policies, key=lambda p: sum(r["losses"][p] / r["robots"] for r in records))
    rows = []
    for (group, _), windows in sorted(by_state.items()):
        if selector.default_window not in windows:
            continue
        default = windows[selector.default_window]
        robots, losses = default["robots"], default["losses"]
        optimum = min(v / robots for w in windows.values() for v in w["losses"].values())
        predicted = selector.predictions(default["features"])
        raw = min(policies, key=lambda p: (predicted[p], policies.index(p)))
        true_order = sorted(policies, key=lambda p: losses[p])
        current = _current_policy(default, policies)
        features = {w: r["features"] for w, r in windows.items()}
        if selector.adaptive and set(selector.windows) - set(features):
            continue
        chosen, hold, info = selector.choose(features, current)
        chosen_loss = windows.get(hold, default)["losses"][chosen] / robots
        rows.append({"group": group, "city": default["city"],
                     "raw_argmin_regret": losses[raw] / robots - optimum,
                     "rule_regret": chosen_loss - optimum,
                     "stay_regret": (losses[current] / robots - optimum) if current else math.nan,
                     "best_fixed_regret": losses[best_fixed] / robots - optimum,
                     "top1": float(raw == true_order[0]),
                     "spearman": _spearman([predicted[p] for p in policies], [losses[p] for p in policies]),
                     "switched": float(current is not None and chosen != current),
                     "vetoed": float(info["switch_vetoed"]), "holding_min": hold})
    if not rows:
        return {"rows": 0}
    counts = Counter(r["group"] for r in rows)

    def mean(key, subset=rows, episode=False):
        values = [(r[key], 1 / counts[r["group"]] if episode else 1.) for r in subset
                  if not math.isnan(r[key])]
        total = sum(w for _, w in values)
        return sum(v * w for v, w in values) / total if total else math.nan

    keys = ("raw_argmin_regret", "rule_regret", "stay_regret", "best_fixed_regret",
            "top1", "spearman", "switched", "vetoed")
    report = {"rows": len(rows), "episodes": len(counts), "best_fixed_policy_on_these_rows": best_fixed,
              "per_row_mean": {k: mean(k) for k in keys},
              "per_episode_mean": {k: mean(k, episode=True) for k in keys},
              "per_city_episode_mean": {city: {k: mean(k, [r for r in rows if r["city"] == city], True)
                                               for k in keys}
                                        for city in sorted({r["city"] for r in rows})},
              "holding_distribution": dict(Counter(str(r["holding_min"]) for r in rows)),
              "units": "regret in loss per robot over the label horizon; lower is better",
              "note": "offline decision diagnostics; actual online validation (validate mode) is still required"}
    return report


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return math.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def build_provenance(records, **extra):
    """Which data informed a selector, split into fitting and held-out validation.

    `scenario_*` lists everything used (excluded from final tests). `fit_*` and
    `validation_scenario_hashes` let validate mode run the selector online on
    genuinely held-out validation episodes only.
    """
    def split(name, key):
        return sorted({key(r) for r in records if r["split"] == name})
    return {
        "scenario_hashes": sorted({r["scenario_sha256"] for r in records}),
        "scenario_workload_hashes": sorted({r["scenario_provenance"]["workload_sha256"] for r in records}),
        "scenario_seeds": sorted({r["scenario_provenance"]["seed"] for r in records}),
        "fit_scenario_hashes": split("train", lambda r: r["scenario_sha256"]),
        "fit_scenario_workload_hashes": split("train", lambda r: r["scenario_provenance"]["workload_sha256"]),
        "fit_scenario_seeds": split("train", lambda r: r["scenario_provenance"]["seed"]),
        "validation_scenario_hashes": split("validation", lambda r: r["scenario_sha256"]),
        "source_identity": records[0]["source_identity"], "records": len(records),
        "label_definition": records[0]["label_definition"],
        "feature_identity": records[0]["feature_identity"], **extra}


def fit_selector(records, *, kind="trees", seed=42, default_window=15., adaptive=False,
                 minimum_gain_fraction=.05, max_iter=200, checkpoint_dir=None, progress=None,
                 target="advantage", margin_scale=1.):
    """Records must have explicit train/validation groups and complete portfolios."""
    if kind not in {"trees", "nn"}:
        raise ValueError("kind must be trees or nn")
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    train = [r for r in records if r["split"] == "train"]
    valid = [r for r in records if r["split"] == "validation"]
    if len(train) < 2 or not valid or len(train) + len(valid) != len(records):
        raise ValueError("need training and validation records; test rows are forbidden")
    train_groups = {r["group"] for r in train}
    if train_groups & {r["group"] for r in valid}:
        raise ValueError("episode leakage between training and validation")
    train_inputs = {r["scenario_sha256"] for r in train}
    if train_inputs & {r["scenario_sha256"] for r in valid}:
        raise ValueError("the same scenario payload occurs in both splits")
    if any("scenario_provenance" in r for r in records):
        for key in ("seed", "workload_sha256"):
            seen = {r["scenario_provenance"][key] for r in train}
            if seen & {r["scenario_provenance"][key] for r in valid}:
                raise ValueError("training/validation share a seed or workload across collection rounds")
    names = tuple(sorted(train[0]["features"]))
    # JSON key sorting must not change the tie-break rule: use portfolio ordering.
    from . import POLICIES
    policies = tuple(p for p in POLICIES if p in train[0]["losses"])
    windows = tuple(sorted({float(r["holding_min"]) for r in train}))
    if default_window not in windows or any(float(r["holding_min"]) not in windows for r in valid):
        raise ValueError("holding windows must be represented in training")
    if (not math.isfinite(minimum_gain_fraction) or minimum_gain_fraction < 0 or max_iter < 1
            or not math.isfinite(margin_scale) or margin_scale < 0):
        raise ValueError("invalid model training settings")
    for row in records:
        if (set(row["features"]) != set(names) or set(row["losses"]) != set(policies)
                or row["robots"] <= 0 or not all(math.isfinite(v) and v >= 0 for v in row["losses"].values())):
            raise ValueError("inconsistent schema or incomplete/non-finite label portfolio")
    x = np.asarray([[r["features"][n] for n in names] for r in train], dtype=float)
    vx = np.asarray([[r["features"][n] for n in names] for r in valid], dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(vx)):
        raise ValueError("non-finite training features")

    def per_robot(rows):
        return np.asarray([[r["losses"][p] / r["robots"] for p in policies] for r in rows])
    loss, vloss = per_robot(train), per_robot(valid)
    if target == "advantage":
        level, vlevel = loss.mean(axis=1), vloss.mean(axis=1)
        heads = {"__level__": (level, vlevel),
                 **{p: (loss[:, i] - level, vloss[:, i] - vlevel) for i, p in enumerate(policies)}}
    else:
        heads = {p: (loss[:, i], vloss[:, i]) for i, p in enumerate(policies)}
    # Equal scenario weight: many correlated snapshots must not outweigh a
    # shorter episode merely because it generated more decision windows.
    counts = Counter(r["group"] for r in train)
    weights = np.asarray([1 / counts[r["group"]] for r in train])
    vcounts = Counter(r["group"] for r in valid)
    vweights = np.asarray([1 / vcounts[r["group"]] for r in valid])
    fit_signature = identity({"records": records, "kind": kind, "seed": seed, "target": target,
                              "max_iter": max_iter, "implementation": digest(__file__),
                              "runtime_versions": runtime_versions()})
    fitted, predictions = {}, {}
    for i, (name, (y, vy)) in enumerate(heads.items()):
        checkpoint = Path(checkpoint_dir) / f"{name.strip('_')}.pickle" if checkpoint_dir else None
        if checkpoint and checkpoint.exists():
            with checkpoint.open("rb") as stream:
                saved = pickle.load(stream)
            if saved["signature"] != fit_signature:
                raise ValueError("incompatible model-fit checkpoint; choose a new output filename")
            fitted[name] = saved["predictor"]
            if progress:
                progress(f"[{i+1}/{len(heads)}] {name}: reused fitted regressor")
        else:
            if progress:
                progress(f"[{i+1}/{len(heads)}] fitting {kind} {target} regressor for {name}")
            if kind == "trees":
                from sklearn.ensemble import HistGradientBoostingRegressor
                predictor = HistGradientBoostingRegressor(max_iter=max_iter, max_leaf_nodes=15,
                    min_samples_leaf=max(2, min(20, len(train) // 10)),
                    l2_regularization=1., early_stopping=False, random_state=seed + i)
                predictor.fit(x, y, sample_weight=weights)
            else:
                predictor = NumpyRegressor(seed=seed + i, epochs=max_iter).fit(x, y, sample_weight=weights)
            fitted[name] = predictor
            if checkpoint:
                atomic_bytes(checkpoint, pickle.dumps({"signature": fit_signature,
                    "predictor": predictor}, protocol=5))
        predictions[name] = fitted[name].predict(vx)
    # Per-policy absolute residuals (reported) and pairwise difference residuals
    # (used as switching margins), weighted by validation episode.
    if target == "advantage":
        absolute = {p: predictions["__level__"] + predictions[p] for p in policies}
    else:
        absolute = {p: predictions[p] for p in policies}
    errors = {p: _weighted_quantile(np.abs(vloss[:, i] - absolute[p]), vweights, .9)
              for i, p in enumerate(policies)}
    pair_errors = {}
    for i, a in enumerate(policies):
        for j, b in enumerate(policies):
            if i < j:
                residual = (absolute[a] - absolute[b]) - (vloss[:, i] - vloss[:, j])
                pair_errors[_pair(a, b)] = _weighted_quantile(np.abs(residual), vweights, .9)
    if not all(math.isfinite(v) for v in [*errors.values(), *pair_errors.values()]):
        raise ValueError("non-finite fitted predictor validation error")
    selector = LossSelector(names, policies, windows, {p: fitted[p] for p in policies}, errors,
                            default_window, minimum_gain_fraction, adaptive, target=target,
                            level_predictor=fitted.get("__level__"), pair_error_q90=pair_errors,
                            margin_scale=margin_scale)
    report = {"training_rows": len(train), "validation_rows": len(valid),
              "training_episodes": len(train_groups), "validation_episodes": len(vcounts),
              "target": target, "kind": kind,
              "validation_error_q90_per_robot": errors,
              "validation_pair_difference_error_q90_per_robot": pair_errors,
              "validation_decisions": evaluate_selector(selector, valid),
              "note": "offline decision diagnostics; run validate mode for actual online episode loss"}
    return selector, report
