"""Joint fully-connected reservation model and its feature/target helpers.

The output is a softmax distribution over importance strata for every robot
type.  This makes every predicted fraction non-negative and guarantees that
fractions sum to one within each type before integer apportionment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence
import json
import math

import numpy as np


@dataclass(frozen=True, slots=True)
class ReservationFeatureSchema:
    """Paper-aligned input ordering for the heterogeneous FCNN.

    The paper uses C class proportions plus the total expected requests per
    vehicle.  Heterogeneity changes the output to K x C; it does not duplicate
    the workload input for every robot type.
    """

    robot_types: tuple[str, ...]
    importance_levels: tuple[float, ...]

    @property
    def names(self) -> tuple[str, ...]:
        names = [
            f"predicted_request_share.importance_{value:g}"
            for value in self.importance_levels
        ]
        names.append("predicted_total_requests_per_robot")
        return tuple(names)


def build_reservation_features(
    schema: ReservationFeatureSchema,
    *,
    predicted_requests_by_importance: Mapping[float, float],
    fleet_count: int,
) -> np.ndarray:
    """Build the paper's C normalized counts plus total requests per robot."""

    if fleet_count <= 0:
        raise ValueError("fleet_count must be positive")
    levels = schema.importance_levels
    predicted = np.asarray(
        [float(predicted_requests_by_importance.get(level, 0.0)) for level in levels]
    )
    if np.any(predicted < 0) or not np.all(np.isfinite(predicted)):
        raise ValueError("predicted request counts must be finite and non-negative")
    total = float(np.sum(predicted))
    shares = predicted / total if total > 0 else np.zeros_like(predicted)
    return np.concatenate((shares, np.asarray([total / float(fleet_count)])))


class ReservationFCNN:
    """Small joint FCNN trained with cross-entropy and Adam."""

    def __init__(
        self,
        input_dim: int,
        robot_types: Sequence[str],
        importance_levels: Sequence[float],
        *,
        hidden_dim: int | None = None,
        seed: int = 42,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.robot_types = tuple(str(value) for value in robot_types)
        self.importance_levels = tuple(float(value) for value in importance_levels)
        if not self.robot_types or not self.importance_levels:
            raise ValueError("robot types and importance levels cannot be empty")
        if len(set(self.robot_types)) != len(self.robot_types):
            raise ValueError("robot types must be unique")
        if tuple(sorted(self.importance_levels)) != self.importance_levels:
            raise ValueError("importance levels must be sorted ascending")
        if (
            len(set(self.importance_levels)) != len(self.importance_levels)
            or any(value <= 0 or not math.isfinite(value) for value in self.importance_levels)
        ):
            raise ValueError("importance levels must be unique, finite and positive")
        self.output_dim = len(self.robot_types) * len(self.importance_levels)
        self.hidden_dim = (
            len(self.importance_levels) if hidden_dim is None else int(hidden_dim)
        )
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, math.sqrt(2.0 / self.input_dim), (self.input_dim, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.w2 = rng.normal(0.0, math.sqrt(2.0 / self.hidden_dim), (self.hidden_dim, self.output_dim))
        self.b2 = np.zeros(self.output_dim)
        self.feature_mean = np.zeros(self.input_dim)
        self.feature_scale = np.ones(self.input_dim)

    def _check_x(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"features must have shape (n, {self.input_dim})")
        if not np.all(np.isfinite(x)):
            raise ValueError("features must be finite")
        return x

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        shaped = logits.reshape(len(logits), len(self.robot_types), len(self.importance_levels))
        shifted = shaped - np.max(shaped, axis=2, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=2, keepdims=True)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        normalized = (x - self.feature_mean) / self.feature_scale
        hidden = np.tanh(normalized @ self.w1 + self.b1)
        probabilities = self._softmax(hidden @ self.w2 + self.b2)
        return hidden, probabilities

    def predict_array(self, features: np.ndarray) -> np.ndarray:
        """Return shape ``(n, robot_types, importance_levels)`` fractions."""

        x = self._check_x(features)
        return self._forward(x)[1]

    def predict(self, features: np.ndarray) -> dict[str, tuple[float, ...]]:
        row = self.predict_array(features)
        if len(row) != 1:
            raise ValueError("predict expects one feature vector; use predict_array for batches")
        return {
            robot_type: tuple(float(value) for value in row[0, index])
            for index, robot_type in enumerate(self.robot_types)
        }

    def loss(self, features: np.ndarray, targets: np.ndarray) -> float:
        x = self._check_x(features)
        y = self._check_targets(targets, len(x))
        probabilities = self._forward(x)[1]
        return float(-np.mean(np.sum(y * np.log(np.maximum(probabilities, 1e-12)), axis=(1, 2))))

    def _check_targets(self, targets: np.ndarray, rows: int) -> np.ndarray:
        y = np.asarray(targets, dtype=float)
        expected = (rows, len(self.robot_types), len(self.importance_levels))
        if y.shape != expected:
            raise ValueError(f"targets must have shape {expected}")
        if np.any(y < 0) or not np.all(np.isfinite(y)):
            raise ValueError("targets must be finite and non-negative")
        if not np.allclose(np.sum(y, axis=2), 1.0, atol=1e-8, rtol=0.0):
            raise ValueError("target fractions must sum to one for each robot type")
        return y

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        epochs: int = 500,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-5,
        verbose: bool = False,
    ) -> list[float]:
        """Fit fractional targets and return the per-epoch training loss."""

        x = self._check_x(features)
        y = self._check_targets(targets, len(x))
        if epochs <= 0 or learning_rate <= 0 or weight_decay < 0:
            raise ValueError("invalid training hyperparameters")
        self.feature_mean = np.mean(x, axis=0)
        scale = np.std(x, axis=0)
        self.feature_scale = np.where(scale > 1e-12, scale, 1.0)
        normalized = (x - self.feature_mean) / self.feature_scale

        parameters = [self.w1, self.b1, self.w2, self.b2]
        first = [np.zeros_like(value) for value in parameters]
        second = [np.zeros_like(value) for value in parameters]
        history: list[float] = []
        beta1, beta2 = 0.9, 0.999

        for step in range(1, int(epochs) + 1):
            hidden = np.tanh(normalized @ self.w1 + self.b1)
            probabilities = self._softmax(hidden @ self.w2 + self.b2)
            data_loss = -np.mean(
                np.sum(y * np.log(np.maximum(probabilities, 1e-12)), axis=(1, 2))
            )
            history.append(float(data_loss))

            grad_logits = (probabilities - y).reshape(len(x), self.output_dim) / len(x)
            grad_w2 = hidden.T @ grad_logits + weight_decay * self.w2
            grad_b2 = np.sum(grad_logits, axis=0)
            grad_hidden = (grad_logits @ self.w2.T) * (1.0 - hidden * hidden)
            grad_w1 = normalized.T @ grad_hidden + weight_decay * self.w1
            grad_b1 = np.sum(grad_hidden, axis=0)
            gradients = [grad_w1, grad_b1, grad_w2, grad_b2]

            for index, (parameter, gradient) in enumerate(zip(parameters, gradients, strict=True)):
                first[index] = beta1 * first[index] + (1.0 - beta1) * gradient
                second[index] = beta2 * second[index] + (1.0 - beta2) * gradient * gradient
                first_hat = first[index] / (1.0 - beta1**step)
                second_hat = second[index] / (1.0 - beta2**step)
                parameter -= learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)
            if verbose and (step == 1 or step % 100 == 0 or step == epochs):
                print(f"epoch={step} loss={history[-1]:.6f}")
        return history

    def save(self, path: str | Path) -> None:
        metadata = json.dumps(
            {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "robot_types": self.robot_types,
                "importance_levels": self.importance_levels,
            }
        )
        np.savez_compressed(
            Path(path), metadata=np.asarray(metadata), w1=self.w1, b1=self.b1,
            w2=self.w2, b2=self.b2, feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ReservationFCNN":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            model = cls(
                metadata["input_dim"], metadata["robot_types"],
                metadata["importance_levels"], hidden_dim=metadata["hidden_dim"],
            )
            for name in ("w1", "b1", "w2", "b2", "feature_mean", "feature_scale"):
                setattr(model, name, np.asarray(data[name], dtype=float).copy())
        return model


class FixedReservationModel:
    """Constant alpha matrix used for perfect-information target evaluation."""

    def __init__(
        self,
        robot_types: Sequence[str],
        importance_levels: Sequence[float],
        fractions_by_type: Mapping[str, Sequence[float]],
    ) -> None:
        self.robot_types = tuple(str(value) for value in robot_types)
        self.importance_levels = tuple(float(value) for value in importance_levels)
        self.input_dim = len(self.importance_levels) + 1
        if len(set(self.robot_types)) != len(self.robot_types):
            raise ValueError("robot types must be unique")
        if (
            tuple(sorted(self.importance_levels)) != self.importance_levels
            or len(set(self.importance_levels)) != len(self.importance_levels)
            or any(value <= 0 or not math.isfinite(value) for value in self.importance_levels)
        ):
            raise ValueError("importance levels must be sorted, unique, finite and positive")
        if set(fractions_by_type) != set(self.robot_types):
            raise ValueError("fixed fractions must cover exactly the model robot types")
        self._fractions: dict[str, tuple[float, ...]] = {}
        for robot_type in self.robot_types:
            row = tuple(float(value) for value in fractions_by_type[robot_type])
            if len(row) != len(self.importance_levels):
                raise ValueError("every fixed fraction row must match importance levels")
            if any(value < 0 or not math.isfinite(value) for value in row):
                raise ValueError("fixed fractions must be finite and non-negative")
            if not math.isclose(sum(row), 1.0, rel_tol=0.0, abs_tol=1e-8):
                raise ValueError("fixed fractions for each type must sum to one")
            self._fractions[robot_type] = row

    def predict(self, features: np.ndarray) -> dict[str, tuple[float, ...]]:
        values = np.asarray(features, dtype=float)
        if values.shape != (self.input_dim,) or not np.all(np.isfinite(values)):
            raise ValueError(f"features must have shape ({self.input_dim},) and be finite")
        return dict(self._fractions)

    @classmethod
    def load(cls, path: str | Path) -> "FixedReservationModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            payload["robot_types"],
            payload["importance_levels"],
            payload["fractions_by_type"],
        )


def select_perfect_information_target(
    candidates: Sequence[np.ndarray],
    objective: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, float]:
    """Select the candidate reservation matrix with minimum realized loss.

    This is the offline target-generation primitive: run each candidate on a
    fully known training scenario and use the minimum-loss candidate as the
    supervised FCNN target.  Ties are resolved by input order.
    """

    if not candidates:
        raise ValueError("at least one candidate reservation matrix is required")
    best: np.ndarray | None = None
    best_loss = math.inf
    for candidate in candidates:
        matrix = np.asarray(candidate, dtype=float)
        value = float(objective(matrix))
        if not math.isfinite(value):
            continue
        if value < best_loss:
            best, best_loss = matrix.copy(), value
    if best is None:
        raise ValueError("all candidate objectives were non-finite")
    return best, best_loss
