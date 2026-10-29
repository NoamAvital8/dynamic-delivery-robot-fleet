"""Action-cost XGBoost and a 64/32-unit NumPy MLP. No fitting on import."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from . import LABEL_VERSION
from .config import POLICIES, guard_city, identity, read, sha, validate_policies
from .provenance import validate_row


def nn_loss_gradients(x, y, weights, biases, sample_weights):
    """Weighted batch objective/derivatives, independently testable without fitting."""
    import numpy as np
    activations = [x]
    for w, b in zip(weights[:2], biases[:2]):
        activations.append(np.maximum(0, activations[-1] @ w + b))
    prediction = activations[-1] @ weights[2] + biases[2]
    error = prediction - y
    denominator = len(x) * y.shape[1]
    loss = float((error**2 * sample_weights[:, None]).sum() / denominator
                 + .5e-4 * sum((w**2).sum() for w in weights))
    derivative = 2 * error * sample_weights[:, None] / denominator
    gradients = [None] * 6
    for layer in (2, 1, 0):
        gradients[2*layer] = activations[layer].T @ derivative + 1e-4 * weights[layer]
        gradients[2*layer+1] = derivative.sum(axis=0)
        if layer:
            derivative = (derivative @ weights[layer].T) * (activations[layer] > 0)
    return loss, gradients


def check_rows(rows, policies=POLICIES):
    policies = validate_policies(policies)
    if not rows:
        raise ValueError("No training states")
    names = sorted(rows[0]["features"])
    if not names:
        raise ValueError("No observable state features")
    seen = set()
    for row in rows:
        guard_city(row["city"], row["split"])
        if row["split"] != "train" or row["label_version"] != LABEL_VERSION:
            raise ValueError("Only compatible training labels may enter fitting/scaling")
        if row["id"] in seen:
            raise ValueError("Duplicate training state")
        seen.add(row["id"])
        if sorted(row["features"]) != names or set(row["costs"]) != set(policies) or row["robots"] < 1:
            raise ValueError("Incomplete feature/action schema")
        if any(word in key.lower() for key in names for word in ("seed", "future", "scenario_id", "city_id")):
            raise ValueError("Forbidden hidden-input feature")
        validate_row(row)
    return names


def fit(rows, family, threads=4, policies=POLICIES):
    import numpy as np
    policies = validate_policies(policies)
    names = check_rows(rows, policies)
    if family not in ("xgboost", "nn"):
        raise ValueError("Unsupported model family")
    if any(r["family"] != family for r in rows):
        raise ValueError("A learner must use its own model-driven trajectory labels")
    x = np.array([[r["features"][k] for k in names] for r in rows], dtype=float)
    loss = np.array([[r["costs"][p] / r["robots"] for p in policies] for r in rows], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(loss).all() or (loss < 0).any():
        raise ValueError("Non-finite features or invalid costs")
    counts = Counter(r["episode"] for r in rows)
    weight = np.array([1. / counts[r["episode"]] for r in rows])
    weight /= weight.sum()
    mean = (x * weight[:, None]).sum(axis=0)
    scale = np.sqrt(((x - mean)**2 * weight[:, None]).sum(axis=0))
    scale = np.where(scale > 1e-8, scale, 1.)
    x = (x - mean) / scale
    target = loss - loss.mean(axis=1, keepdims=True)
    common = {"kind": family, "label_version": LABEL_VERSION, "policies": list(policies),
              "feature_names": names, "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
              "training_cities": sorted({r["city"] for r in rows}),
              "training_episodes": sorted(counts), "training_states": len(rows),
              "training_scenario_hashes": sorted({r["scenario_sha256"] for r in rows}),
              "training_seeds": sorted({r["seed"] for r in rows}),
              "training_records_sha256": identity(sorted((r["id"], r["content_sha256"]) for r in rows)),
              "target": "per_robot_cost_minus_state_portfolio_mean"}
    if family == "xgboost":
        import xgboost as xgb
        heads = []
        for index in range(len(policies)):
            model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=.04,
                min_child_weight=8, reg_lambda=10., subsample=1., colsample_bytree=1.,
                objective="reg:squarederror", tree_method="hist", n_jobs=threads, random_state=710 + index)
            # Mean-one sample weights preserve the meaning of tree regularization.
            model.fit(x, target[:, index], sample_weight=weight * len(weight))
            heads.append(model.get_booster().save_raw(raw_format="json").decode())
        return {**common, "boosters": heads, "library_version": xgb.__version__}
    rng = np.random.default_rng(71)
    widths = (x.shape[1], 64, 32, len(policies))
    weights = [rng.normal(0, np.sqrt(2 / widths[i]), (widths[i], widths[i+1])) for i in range(3)]
    biases = [np.zeros(widths[i+1]) for i in range(3)]
    ymean = (target * weight[:, None]).sum(axis=0)
    yscale = max(1e-6, float(np.sqrt(((target - ymean)**2 * weight[:, None]).sum() / len(policies))))
    y = (target - ymean) / yscale
    params = [item for pair in zip(weights, biases) for item in pair]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    tick = 0
    for _ in range(60):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 512):
            ii = order[start:start+512]
            _, gradients = nn_loss_gradients(x[ii], y[ii], weights, biases, weight[ii] * len(weight))
            tick += 1
            for j, (p, gradient) in enumerate(zip(params, gradients)):
                gradient = np.clip(gradient, -10., 10.)
                first[j] = .9 * first[j] + .1 * gradient
                second[j] = .999 * second[j] + .001 * gradient**2
                p -= .001 * (first[j] / (1 - .9**tick)) / (np.sqrt(second[j] / (1 - .999**tick)) + 1e-8)
    return {**common, "weights": [w.tolist() for w in weights], "biases": [b.tolist() for b in biases],
            "target_mean": ymean.tolist(), "target_scale": yscale,
            "architecture": [64, 32], "epochs": 60}


class Predictor:
    def __init__(self, payload, expected_policies=None):
        self.payload = payload
        self.policies = validate_policies(payload["policies"])
        if (payload.get("label_version") != LABEL_VERSION
                or (expected_policies is not None and self.policies != tuple(expected_policies))):
            raise ValueError("Model/collector scientific identity mismatch")
        if payload["kind"] == "bootstrap" and payload["policy"] not in self.policies:
            raise ValueError("Bootstrap action is outside the model portfolio")
        self.heads = []
        if payload["kind"] == "nn":
            import numpy as np
            self.weights = [np.asarray(w, dtype=float) for w in payload["weights"]]
            self.biases = [np.asarray(b, dtype=float) for b in payload["biases"]]
            if (len(self.weights) != 3 or len(self.biases) != 3
                    or self.weights[-1].shape[-1] != len(self.policies)
                    or self.biases[-1].shape != (len(self.policies),)
                    or len(payload["target_mean"]) != len(self.policies)):
                raise ValueError("Neural output schema does not match the action portfolio")
        if payload["kind"] == "xgboost":
            if len(payload["boosters"]) != len(self.policies):
                raise ValueError("Tree output schema does not match the action portfolio")
            import xgboost as xgb
            for value in payload["boosters"]:
                head = xgb.Booster(params={"nthread": 1})
                head.load_model(bytearray(value.encode()))
                head.set_param({"nthread": 1})
                self.heads.append(head)

    @classmethod
    def load(cls, path, expected_hash=None, expected_policies=None):
        if expected_hash is not None and sha(path) != expected_hash:
            raise ValueError("Published model hash changed")
        return cls(read(path), expected_policies)

    def choose(self, features):
        if self.payload["kind"] == "bootstrap":
            return self.payload["policy"]
        import numpy as np
        p = self.payload
        if set(features) != set(p["feature_names"]):
            raise ValueError("Inference feature schema changed")
        x = np.array([[features[k] for k in p["feature_names"]]], dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Non-finite inference state")
        x = (x - p["feature_mean"]) / p["feature_scale"]
        if p["kind"] == "xgboost":
            costs = np.array([head.inplace_predict(x)[0] for head in self.heads])
        elif p["kind"] == "nn":
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                x = x @ w + b
                if i < 2:
                    x = np.maximum(0, x)
            costs = x[0] * p["target_scale"] + p["target_mean"]
        else:
            raise ValueError("Unknown serialized predictor")
        if not np.isfinite(costs).all():
            raise ValueError("Non-finite predicted costs")
        return self.policies[int(np.argmin(costs))]
