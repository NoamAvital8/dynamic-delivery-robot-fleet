import networkx as nx

from delivery_fleet.charging import (
    DEFAULT_CHARGING_POWER_W,
    DEFAULT_NUMBER_OF_PORTS,
    ChargingConfig,
    add_charging_stations,
    select_charging_station_nodes,
)


def weighted_path_graph(n: int) -> nx.Graph:
    graph = nx.path_graph(n)
    nx.set_edge_attributes(graph, 100.0, "length")
    return graph


def test_station_selection_is_reproducible_for_same_seed() -> None:
    graph = weighted_path_graph(20)
    config = ChargingConfig(station_count=5, seed=123, placement_fraction=0.75)

    first = select_charging_station_nodes(graph, config)
    second = select_charging_station_nodes(graph, config)

    assert first == second
    assert len(first) == 5
    assert len(set(first)) == 5


def test_different_seed_can_change_random_first_station() -> None:
    graph = weighted_path_graph(100)

    a = select_charging_station_nodes(
        graph, ChargingConfig(station_count=1, seed=1)
    )
    b = select_charging_station_nodes(
        graph, ChargingConfig(station_count=1, seed=2)
    )

    assert a != b


def test_add_charging_stations_annotates_selected_nodes() -> None:
    graph = weighted_path_graph(30)
    result, stations = add_charging_stations(
        graph,
        ChargingConfig(station_count=4, seed=42),
    )

    selected = {station.node_id for station in stations}
    assert len(selected) == 4
    assert all(result.nodes[node]["is_charging_station"] for node in selected)
    assert sum(
        bool(data.get("is_charging_station"))
        for _, data in result.nodes(data=True)
    ) == 4

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
    assert all("is_charging_station" not in data for _, data in graph.nodes(data=True))
