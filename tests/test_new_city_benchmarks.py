import json
from pathlib import Path
import sys

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_new_city_benchmarks import NEW_CITIES, check_suite, prepare_city, validate_graph


def test_chargers_prepared_once_without_modifying_source(tmp_path,monkeypatch):
    # Other runner integration tests install in-memory graph-reader wrappers.
    # This preprocessing test must exercise the unmodified physical file reader.
    monkeypatch.setattr(nx,'read_graphml',nx.readwrite.graphml.read_graphml)
    source=tmp_path/'source';source.mkdir()
    prepared=tmp_path/'prepared'
    graph=nx.path_graph(4)
    for node in graph:
        graph.nodes[node].update(x=28.+node*.001,y=-26.,in_cluster=0,is_cluster_representative=node==0)
    nx.set_edge_attributes(graph,100.,'length')
    path=source/'johannesburg.graphml';nx.write_graphml(graph,path)
    original=path.read_bytes()
    first=prepare_city(('johannesburg',str(source),str(prepared),'code_v1'))
    second=prepare_city(('johannesburg',str(source),str(prepared),'code_v1'))
    assert first==second and first['charging_stations']>0
    assert path.read_bytes()==original
    assert not any(a.get('is_charging_station') for _,a in nx.read_graphml(path,node_type=int).nodes(data=True))
    with pytest.raises(RuntimeError,match='incompatible'):
        prepare_city(('johannesburg',str(source),str(prepared),'code_v2'))


def test_invalid_cluster_partition_rejected():
    graph=nx.path_graph(3)
    for node in graph:graph.nodes[node].update(x=0.,y=0.,in_cluster=0)
    nx.set_edge_attributes(graph,10.,'length')
    with pytest.raises(ValueError,match='representative'):validate_graph(graph)


def test_suite_rejects_original_city(tmp_path):
    path=tmp_path/'suite.json'
    data={'cities':{c:{'test':[{}]*5} for c in NEW_CITIES}}
    path.write_text(json.dumps(data));assert check_suite(path)==data
    data['cities']['tel_aviv']={'test':[{}]*5};path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='ONLY'):check_suite(path)
