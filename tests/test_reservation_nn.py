from __future__ import annotations

import numpy as np

from delivery_fleet.reservation_nn import (
    ReservationFCNN,
    ReservationFeatureSchema,
    build_reservation_features,
    select_perfect_information_target,
)


def test_joint_fcnn_trains_and_outputs_one_simplex_per_robot_type(tmp_path) -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 5))
    y = np.zeros((80, 2, 3))
    high = x[:, 0] > 0
    y[high, :, 2] = 1.0
    y[~high, :, 0] = 1.0

    model = ReservationFCNN(5, ("a", "b"), (1.0, 2.0, 5.0), hidden_dim=10, seed=4)
    initial = model.loss(x, y)
    history = model.fit(x, y, epochs=300, learning_rate=0.02)
    predicted = model.predict_array(x)
    assert history[-1] < initial * 0.25
    np.testing.assert_allclose(predicted.sum(axis=2), 1.0, atol=1e-12)
    assert np.all(predicted >= 0)

    path = tmp_path / "reservation_model.npz"
    model.save(path)
    restored = ReservationFCNN.load(path)
    np.testing.assert_allclose(restored.predict_array(x), predicted)


def test_feature_builder_has_stable_schema_and_finite_values() -> None:
    schema = ReservationFeatureSchema(("a", "b"), (1.0, 5.0))
    features = build_reservation_features(
        schema,
        demand_rate_per_minute={1.0: 2.0, 5.0: 1.0},
        fleet_count=4,
        backlog_by_importance={1.0: 3, 5.0: 1},
        busy_fraction_by_type={"a": 0.5, "b": 1.0},
        mean_battery_fraction_by_type={"a": 0.8, "b": 0.4},
        cluster_rate_per_minute={(0, 1.0): 2.0, (1, 5.0): 1.0},
        charger_congestion=0.25,
        remaining_horizon_fraction=0.75,
    )
    assert features.shape == (len(schema.names),)
    assert np.all(np.isfinite(features))
    np.testing.assert_allclose(features[:2], (2.0 / 3.0, 1.0 / 3.0))


def test_perfect_information_target_uses_minimum_realized_loss() -> None:
    candidates = [np.asarray([[1.0, 0.0]]), np.asarray([[0.0, 1.0]])]
    chosen, loss = select_perfect_information_target(
        candidates, lambda candidate: 8.0 if candidate[0, 0] else 3.0
    )
    np.testing.assert_array_equal(chosen, candidates[1])
    assert loss == 3.0
