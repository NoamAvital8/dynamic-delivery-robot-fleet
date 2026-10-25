from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import networkx as nx
import pytest
from requests import Response
from requests.exceptions import ConnectTimeout, HTTPError


@pytest.fixture
def downloader(monkeypatch):
    # Exercise the downloader without contacting OSM or installing its GIS stack.
    fake_ox = SimpleNamespace(
        settings=SimpleNamespace(use_cache=False),
        convert=SimpleNamespace(to_undirected=lambda graph: graph.to_undirected()),
        save_graphml=lambda graph, path: nx.write_graphml(graph, path),
    )
    monkeypatch.setitem(sys.modules, "osmnx", fake_ox)
    path = Path(__file__).resolve().parents[1] / "scripts/download_osm_graph.py"
    spec = importlib.util.spec_from_file_location("download_osm_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, fake_ox


def test_network_retry_preserves_completed_graph_and_resume_skips_download(
    downloader, tmp_path, monkeypatch,
):
    module, ox = downloader
    calls = []
    delays = []

    def fetch(*args, **kwargs):
        calls.append(args)
        if len(calls) < 3:
            raise ConnectTimeout("temporary Overpass outage")
        graph = nx.path_graph(3).to_directed()
        graph.add_node(99)  # The isolated component must still be removed.
        return graph

    ox.graph_from_place = fetch
    monkeypatch.setattr(module.time, "sleep", delays.append)
    result = module.download_graph("haifa", tmp_path, retry_delay_seconds=1)
    assert len(calls) == 3
    assert delays == [1, 2]
    assert ox.settings.use_cache is True
    assert result["nodes"] == 3
    assert nx.is_connected(nx.read_graphml(tmp_path / "haifa.graphml"))
    assert not (tmp_path / "haifa.graphml.tmp").exists()
    assert module.download_graph("haifa", tmp_path, resume=True) == result
    assert len(calls) == 3


def test_retry_limit_keeps_original_network_error(downloader, tmp_path, monkeypatch):
    module, ox = downloader
    calls = []

    def fetch(*args, **kwargs):
        calls.append(1)
        raise ConnectTimeout("still unavailable")

    ox.graph_from_place = fetch
    monkeypatch.setattr(module.time, "sleep", lambda delay: None)
    with pytest.raises(ConnectTimeout, match="still unavailable"):
        module.download_graph("haifa", tmp_path, attempts=2)
    assert len(calls) == 2
    assert not (tmp_path / "haifa.json").exists()


def test_permanent_http_failure_is_not_retried(downloader, tmp_path):
    module, ox = downloader
    calls = []

    def fetch(*args, **kwargs):
        calls.append(1)
        response = Response()
        response.status_code = 400
        raise HTTPError("invalid request", response=response)

    ox.graph_from_place = fetch
    with pytest.raises(HTTPError):
        module.download_graph("haifa", tmp_path)
    assert len(calls) == 1


def test_all_city_failure_keeps_partial_summary(downloader, tmp_path, monkeypatch):
    module, _ = downloader

    def download(city, output_dir, **kwargs):
        if city == "haifa":
            raise ConnectTimeout("offline")
        return {"key": city}

    monkeypatch.setattr(module, "download_graph", download)
    monkeypatch.setattr(sys, "argv", ["download_osm_graph.py", "all", "--output-dir", str(tmp_path)])
    with pytest.raises(ConnectTimeout):
        module.main()
    assert json.loads((tmp_path / "summary.json").read_text()) == [{"key": "tel_aviv"}]
