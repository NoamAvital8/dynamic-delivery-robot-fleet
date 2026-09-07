import networkx as nx

from delivery_fleet.charging import (
    DEFAULT_CHARGING_POWER_W,
    DEFAULT_NUMBER_OF_PORTS,
    ChargingConfig,
    add_charging_stations,
    max_distance_to_nearest_station,
    select_charging_station_nodes,
)
from delivery_fleet.defaults import (
    MAX_DISTANCE_TO_CHARGING_STATION_M,
    MIN_ROBOT_FULL_BATTERY_RANGE_M,
)


def weighted_path_graph(n: int) -> nx.Graph:
    graph = nx.path_graph(n)
    nx.set_edge_attributes(graph, 100.0, "length")
    return graph


def test_station_selection_is_reproducible_for_same_seed() -> None:
    graph = weighted_path_graph(100)
    config = ChargingConfig(
        max_distance_to_station_m=500.0,
        seed=123,
        placement_fraction=0.75,
    )

    first = select_charging_station_nodes(graph, config)
    second = select_charging_station_nodes(graph, config)

    assert first == second
    assert len(set(first)) == len(first)
    assert max_distance_to_nearest_station(graph, first) <= 500.0


def test_different_seed_can_change_random_first_station() -> None:
    graph = weighted_path_graph(100)

    # A huge coverage radius means placement stops after the random first node,
    # making the effect of the seed directly observable.
    a = select_charging_station_nodes(
        graph,
        ChargingConfig(max_distance_to_station_m=20_000.0, seed=1),
    )
    b = select_charging_station_nodes(
        graph,
        ChargingConfig(max_distance_to_station_m=20_000.0, seed=2),
    )

    assert len(a) == len(b) == 1
    assert a != b


def test_selection_runs_until_coverage_threshold_is_satisfied() -> None:
    graph = weighted_path_graph(80)
    config = ChargingConfig(max_distance_to_station_m=700.0, seed=42)

    stations = select_charging_station_nodes(graph, config)

    assert max_distance_to_nearest_station(graph, stations) <= 700.0


def test_default_coverage_matches_minimum_robot_range_safety_rule() -> None:
    assert MAX_DISTANCE_TO_CHARGING_STATION_M == 2_000.0
    assert MIN_ROBOT_FULL_BATTERY_RANGE_M == 4_000.0
    assert (
        2 * MAX_DISTANCE_TO_CHARGING_STATION_M
        <= MIN_ROBOT_FULL_BATTERY_RANGE_M
    )


def test_add_charging_stations_annotates_selected_nodes() -> None:
    graph = weighted_path_graph(80)
    config = ChargingConfig(max_distance_to_station_m=700.0, seed=42)
    result, stations = add_charging_stations(graph, config)

    selected = {station.node_id for station in stations}
    assert len(selected) == len(stations)
    assert all(result.nodes[node]["is_charging_station"] for node in selected)
    assert sum(
        bool(data.get("is_charging_station"))
        for _, data in result.nodes(data=True)
    ) == len(stations)
    assert max_distance_to_nearest_station(result, list(selected)) <= 700.0

    # All charging stations have the same fixed V1 capabilities.
    assert all(
        station.charging_power_w == DEFAULT_CHARGING_POWER_W
        for station in stations
    )
    assert all(
        station.number_of_ports == DEFAULT_NUMBER_OF_PORTS
        for station in stations
    )
    assert all(
        result.nodes[node]["charging_power_w"] == DEFAULT_CHARGING_POWER_W
        for node in selected
    )
    assert all(
        result.nodes[node]["charging_ports"] == DEFAULT_NUMBER_OF_PORTS
        for node in selected
    )

    # Input graph is unchanged by default.
    assert all(
        "is_charging_station" not in data
        for _, data in graph.nodes(data=True)
    )
