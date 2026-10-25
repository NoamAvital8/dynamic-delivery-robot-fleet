"""Event-boundary adapter over the existing ten dispatch implementations.

The coordinated runner supplies one physical event engine. Existing candidate
search functions are compiled from each original generated source into that
engine's namespace. Only decision functions, reservation controller and idle
planner change at a switch. No physical state is transplanted between runners.

Generated-source anchors deliberately fail closed when upstream code changes.
No source files in scripts/ are edited. See docs/adaptive_meta_training.md for
the exact switching semantics and fixed-policy equivalence requirements.
"""
from __future__ import annotations

import ast
from collections import defaultdict, deque
import copy
from dataclasses import replace
from functools import partial
import hashlib
import heapq
import io
import json
import math
from pathlib import Path
import pickle
import symtable
import sys
import types

from . import POLICIES

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
MODULE = "delivery_fleet._adaptive_event_engine"
SNAPSHOT_VERSION = 2
# Shared exact-index state of this process: original and accelerated
# ChargerDistanceIndex methods, and the cache folder the next build uses.
_EXACT = {"original": None, "patched": None, "path": None}

MLE_POLICIES = ("queue_mle_reservation", "full_mle_reservation")
SPATIAL_POLICIES = frozenset({"full", "full_no_idle", "full_queue_aware", "full_coordinated_idle"})

# Scientific controller settings of the original runners and their CLI
# defaults. Every value is passed to the shared engine; unknown keys fail.
CONTROLLER_DEFAULTS = {
    "prior_concentration": 4.0,
    "reservation_lookback_min": 100.0,
    "idle_horizon_min": 45.0,
    "idle_replan_interval_min": 15.0,
    "idle_candidate_clusters": 5,
    "idle_candidate_chargers": 3,
    "idle_max_demand_clusters": 32,
    "idle_max_actions_per_epoch": 32,
    "idle_minimum_gain_fraction": 0.001,
    "idle_relocation_uncertainty_penalty": 0.0,
    "idle_switch_gain_fraction": 0.005,
    "idle_retarget_cooldown_min": 2.0,
    "queue_charger_alternatives": 3,
    "queue_charger_scan_limit": 12,
}


def controller_key(policy):
    """Reservation controller used by a policy, or None for no reservation."""
    if policy in MLE_POLICIES:
        return "mle"
    if policy == "paper_sa_adapted":
        return "paper"
    if policy in SPATIAL_POLICIES:
        return "spatial"
    return None


def controller_settings(value=None):
    """Complete, validated controller settings (defaults filled in).

    MLE worker processes are an execution setting and stay fixed at one.
    """
    from delivery_fleet.mle_reservation import MLEConfig
    value = dict(value or {})
    mle = dict(value.pop("mle", None) or {})
    unknown = set(value) - set(CONTROLLER_DEFAULTS)
    if unknown:
        raise ValueError(f"unsupported controller settings: {sorted(unknown)}")
    settings = {}
    for name, default in CONTROLLER_DEFAULTS.items():
        raw = value.get(name, default)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"controller setting {name} must be numeric")
        if isinstance(default, int) and float(raw) != int(raw):
            raise ValueError(f"controller setting {name} must be an integer")
        number = type(default)(raw)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"controller setting {name} must be finite and non-negative")
        settings[name] = number
    fields = [name for name in MLEConfig.__dataclass_fields__ if name != "processes"]
    if set(mle) - set(fields):
        raise ValueError(f"unsupported MLE settings: {sorted(set(mle) - set(fields))}")
    defaults = MLEConfig()
    config = MLEConfig(**{**{name: getattr(defaults, name) for name in fields}, **mle, "processes": 1})
    settings["mle"] = {name: getattr(config, name) for name in fields}
    return settings


def _settings_argv(settings):
    argv = []
    for name in CONTROLLER_DEFAULTS:
        argv += ["--" + name.replace("_", "-"), str(settings[name])]
    for name, value in settings["mle"].items():
        argv += ["--mle-" + name.replace("_", "-"), str(value)]
    return argv


class _EventCounter:
    """Pickle-stable integer tie breaker, including on Python 3.14.

    Python 3.14's itertools.count is not pickleable. Keep the same next integer
    explicitly so taking/restoring a checkpoint never consumes an event id.
    """
    def __init__(self, next_value):
        self.next_value = next_value

    def __iter__(self):
        return self

    def __next__(self):
        value = self.next_value
        self.next_value += 1
        return value


class _InlineIndexExecutor:
    """Existing mmap builder's one-worker case without an extra child process."""
    def __init__(self, *, max_workers, initializer, initargs, **kwargs):
        if max_workers != 1:
            raise ValueError("meta training permits only inline index preparation")
        initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def map(self, function, jobs):
        return map(function, jobs)


def _verified_index_loader(original):
    """Hash a shared index once, then trust it while its files are unchanged.

    The original loader SHA-256-hashes both tables (23 GiB for Beijing) under
    the exclusive build lock on every simulator construction. After one full
    verification, a record of (size, mtime_ns, inode) for the three files is
    kept next to the tables; later loads only re-check that record, the ready
    metadata, dtype and shape. Any replaced or rewritten file is fully hashed
    again. This does not detect silent in-place bit rot.
    """
    import numpy as np
    from .artifacts import atomic_json
    names = ("ready.json", "distances.npy", "predecessors.npy")

    def stats(folder):
        return {name: [s.st_size, s.st_mtime_ns, s.st_ino]
                for name in names for s in [(folder / name).stat()]}

    def load(folder, key, shape):
        folder = Path(folder)
        if not (folder / "ready.json").exists():
            return None
        expected = {"version": 1, "key": key, "shape": [int(v) for v in shape]}
        before = stats(folder)
        try:
            record = json.loads((folder / "meta_verified.json").read_text())
        except (FileNotFoundError, ValueError):
            record = None
        if record == {**expected, "stats": before}:
            meta = json.loads((folder / "ready.json").read_text())
            if meta.get("key") != key or tuple(meta.get("shape", ())) != tuple(shape):
                raise RuntimeError("shared charger index identity mismatch")
            arrays = []
            for name, dtype in (("distances.npy", np.float64), ("predecessors.npy", np.int32)):
                value = np.load(folder / name, mmap_mode="r")
                if value.shape != tuple(shape) or value.dtype != dtype:
                    raise RuntimeError("shared charger index dtype/shape mismatch")
                arrays.append(value)
            return tuple(arrays)
        arrays = original(folder, key, shape)
        if arrays is not None and stats(folder) == before:
            atomic_json(folder / "meta_verified.json", {**expected, "stats": before})
        return arrays
    return load


def _use_shared_index(shared_cache):
    """Select the exact-index cache for the next simulator in this process.

    `None` restores the original in-process index. The accelerated methods are
    installed once; the cache folder can change between simulators (serial
    runs), and nested index-building pools are never created.
    """
    from delivery_fleet.routing import ChargerDistanceIndex as Index
    names = ("_build_dense_if_possible", "distances_to_stations", "path_from_station")
    if _EXACT["original"] is None:
        _EXACT["original"] = {name: getattr(Index, name) for name in names}
    if shared_cache is None:
        for name, method in _EXACT["original"].items():
            setattr(Index, name, method)
        return
    if _EXACT["patched"] is None:
        if str(ROOT / "operations") not in sys.path:
            sys.path.insert(0, str(ROOT / "operations"))
        import exact_runtime
        # Parallelism is at rollout level; never create nested worker pools.
        exact_runtime.ProcessPoolExecutor = _InlineIndexExecutor
        exact_runtime._load_valid = _verified_index_loader(exact_runtime._load_valid)
        original_shared = exact_runtime.shared_dense
        exact_runtime.shared_dense = lambda index, _folder, workers: original_shared(index, _EXACT["path"], workers)
        exact_runtime.install(Path(shared_cache), workers=1)
        _EXACT["patched"] = {name: getattr(Index, name) for name in names}
    for name, method in _EXACT["patched"].items():
        setattr(Index, name, method)
    _EXACT["path"] = Path(shared_cache).resolve()


def sources():
    """Materialize policy programs without invoking any main()."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_nyc_heuristic_policy as h
    import run_nyc_anticipatory_idle_policy as idle
    import run_nyc_queue_aware_policy as queue
    import run_nyc_coordinated_idle_policy as coord
    import run_nyc_mle_reservation_policy as mle
    base = h.corrected.TARGET.read_text(encoding="utf-8")
    loss = h.corrected._patch_source(base)
    heuristic = h._patch_heuristic(loss)
    legacy = idle._patch_idle(heuristic)
    queued = queue._patch_queue(legacy)
    coordinated = coord._patch_coordinated(queued)
    return {
        "myopic_ab": base, "reactive_insertion": loss,
        "paper_sa_adapted": heuristic, "full_no_nn": legacy,
        "full_no_idle": heuristic, "full": legacy,
        "full_queue_aware": queued, "full_coordinated_idle": coordinated,
        "queue_mle_reservation": mle._patch_mle(queued),
        "full_mle_reservation": mle._patch_mle(coordinated),
    }


def _main(tree):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


class _Promote(ast.NodeTransformer):
    """Main's closure cells become globals; helper-local closures stay local."""
    def __init__(self, names):
        self.names = set(names)

    def visit_Nonlocal(self, node):
        promoted = [n for n in node.names if n in self.names]
        remaining = [n for n in node.names if n not in self.names]
        return ([ast.Global(promoted)] if promoted else []) + (
            [ast.Nonlocal(remaining)] if remaining else [])

    def visit_With(self, node):
        # Audit files are optional output, never simulator inputs. Counterfactual
        # branches must not append to one another's queue/idle audit logs.
        if any(isinstance(n, ast.Name) and n.id in {"queue_log_path", "idle_decisions_path"}
               for item in node.items for n in ast.walk(item.context_expr)):
            return ast.Pass()
        return self.generic_visit(node)


class _Breaks(ast.NodeTransformer):
    def __init__(self):
        self.depth = 0

    def visit_For(self, node):
        self.depth += 1
        node = self.generic_visit(node)
        self.depth -= 1
        return node

    visit_While = visit_For

    def visit_Break(self, node):
        if self.depth:
            return node
        return [ast.Assign([ast.Name("_meta_finished", ast.Store())], ast.Constant(True)), node]


def _compile(nodes, filename):
    import __future__
    return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                   filename, "exec", flags=__future__.annotations.compiler_flag)


def _functions(source, namespace):
    tree = ast.parse(source)
    main = _main(tree)
    names = next(t for t in symtable.symtable(source, "policy", "exec").get_children()
                 if t.get_name() == "main").get_locals()
    definitions = {n.name: n for n in main.body if isinstance(n, ast.FunctionDef)}
    # Candidate search is read-only with respect to physical execution. Compile
    # its complete named-helper dependency closure, not a handpicked approximation.
    wanted, pending = set(), ["choose_insertion", "refresh_reservations"]
    while pending:
        name = pending.pop()
        if name in wanted or name not in definitions:
            continue
        wanted.add(name)
        pending.extend(n.func.id for n in ast.walk(definitions[name])
                       if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                       and n.func.id in definitions)
    selected = [_Promote(names).visit(copy.deepcopy(n)) for n in main.body
                if isinstance(n, ast.FunctionDef) and n.name in wanted]
    exec(_compile(selected, "<adaptive decision functions>"), namespace)
    return {name: namespace[name] for name in wanted}


# Only mutable execution state is checkpointed. Static map/routing data and
# transient candidate caches are deliberately excluded. Aliasing is retained
# by a single pickle stream (robots == robot_by_id values == controller robots).
STATE_NAMES = (
    "robots", "robot_by_id", "schedules", "station_state", "charge_token",
    "pending", "metrics", "active_phases", "phase_version", "service_lock",
    "service_token", "delivery_distance_m", "background_distance_m", "events",
    "event_counter", "delivered", "now", "scenario", "idle_intents", "idle_stats",
    "idle_service_finish_min", "idle_last_target_change", "coordinated_pending_times",
    "queue_stats", "queue_predictions_by_robot", "queue_request_predictions",
    "queue_observed_waits", "_meta_finished", "_meta_observed", "_meta_controllers",
    "_meta_demand", "_meta_policy", "B5_STATS", "direct_distance_cache",
    "reservation_assignment", "assignment_queue_forecast",
)

# Diagnostics and bookkeeping that never influence a later decision. They are
# left out of the tail digest so identical physical states can share a label.
# Controllers are digested separately (only the one the current policy uses).
DIGEST_EXCLUDED = frozenset({
    "idle_stats", "queue_stats", "B5_STATS", "queue_observed_waits",
    "queue_predictions_by_robot", "queue_request_predictions",
    "direct_distance_cache", "_meta_controllers",
})

# Physical executors the runtime replaces; no decision closure may contain them.
_RUNTIME_OVERRIDES = frozenset({
    "_meta_observe", "_meta_refresh_arrival", "plan_idle_fleet",
    "request_coordinated_idle_replan", "begin_background_return", "start_idle_action",
})


def _no_refresh(*_, **__):
    return None


class _Writer(pickle.Pickler):
    def __init__(self, stream, refs):
        super().__init__(stream, protocol=5)
        self.refs = {id(value): key for key, value in refs.items()}

    def persistent_id(self, obj):
        return self.refs.get(id(obj))


class _Reader(pickle.Unpickler):
    def __init__(self, stream, refs):
        super().__init__(stream)
        self.refs = refs

    def persistent_load(self, key):
        return self.refs[key]


class FleetRuntime:
    """One shared physical state and ten switchable decision adapters.

    One instance per process. Snapshot pickle files are trusted local artifacts,
    not a format for accepting untrusted downloads.
    """
    def __init__(self, graph, scenario, spatial_model, paper_model, prior_rates,
                 *, scratch, shortlist_k=10, shared_cache=None, settings=None):
        from delivery_fleet.anticipatory_policy import OnlineAnticipatoryReservation
        from delivery_fleet.reservation_nn import ReservationFCNN
        from delivery_fleet.idle_planning import IdleFleetPlanner
        from delivery_fleet.coordinated_idle import CoordinatedIdleFleetPlanner
        from delivery_fleet.spatial_demand import GammaPoissonDemandModel
        from delivery_fleet.charging import DEFAULT_CHARGING_POWER_W

        self.settings = controller_settings(settings)
        program = sources()
        self.source_digest = hashlib.sha256("".join(program[p] for p in POLICIES).encode()
                                            + Path(__file__).read_bytes()).hexdigest()
        from .artifacts import digest, identity
        self.input_digest = identity({"files": [digest(p) for p in
            (graph, scenario, spatial_model, paper_model)], "priors": prior_rates,
            "shortlist_k": shortlist_k, "settings": self.settings})
        source = program["full_mle_reservation"]
        tree = ast.parse(source)
        main = _main(tree)
        main_names = next(t for t in symtable.symtable(source, "engine", "exec").get_children()
                          if t.get_name() == "main").get_locals()
        loops = [(i, n) for i, n in enumerate(main.body)
                 if isinstance(n, ast.While) and isinstance(n.test, ast.Name) and n.test.id == "events"]
        if len(loops) != 1:
            raise RuntimeError("expected one original physical event loop")
        offset, loop = loops[0]
        module = types.ModuleType(MODULE)
        module.__file__ = str(SCRIPTS / "run_nyc_benchmark5.py")
        sys.modules[MODULE] = module
        ns = self.ns = module.__dict__
        top = [n for n in tree.body if n is not main
               and not (isinstance(n, ast.If) and "__name__" in ast.unparse(n.test))]
        exec(_compile(top, "<adaptive original definitions>"), ns)
        _use_shared_index(shared_cache)
        argv = ["meta-runtime", "--graph", str(graph), "--scenario", str(scenario),
                "--output", str(Path(scratch) / "unused.json"), "--idle-processes", "1",
                "--mle-processes", "1", "--shortlist-k", str(shortlist_k),
                "--importance-prior-rates-per-hour", json.dumps(prior_rates),
                *_settings_argv(self.settings)]
        previous = sys.argv
        try:
            sys.argv = argv
            exec(_compile([_Promote(main_names).visit(copy.deepcopy(n))
                           for n in main.body[:offset]], "<adaptive initialize>"), ns)
        finally:
            sys.argv = previous
        # Preserve the original counter's next id exactly. The original counter
        # is discarded; this value is returned by our counter's first next().
        ns["event_counter"] = _EventCounter(next(ns["event_counter"]))
        ns["save_progress"] = lambda *_: None
        ns["print"] = lambda *a, **k: None
        # Source event loop remains the physical authority. Arrivals are observed
        # exactly once by every controller, including inactive ones. The arrival
        # refresh keeps the MLE runner's arrival=True semantics (the new order is
        # not an existing backlog).
        event_body = copy.deepcopy(loop.body)
        dispatch = next(n for n in event_body if isinstance(n, ast.If)
                        and ast.unparse(n.test) == "kind == 'order_arrival'")
        dispatch.body = ast.parse("_meta_observe(payload, now)\npending.append(payload)\n"
                                  "_meta_refresh_arrival(now)\ndispatch_pending(now)").body
        cursor = dispatch
        while cursor.orelse and isinstance(cursor.orelse[0], ast.If):
            cursor = cursor.orelse[0]
            if ast.unparse(cursor.test) == "kind == 'mle_reservation_refresh'":
                cursor.body = [ast.If(ast.parse("_meta_policy in ('queue_mle_reservation', 'full_mle_reservation')",
                                                mode="eval").body, cursor.body, [])]
        event_body = [_Promote(main_names).visit(n) for n in event_body]
        event_body = [_Breaks().visit(n) for n in event_body]
        event_body = [item for n in event_body for item in (n if isinstance(n, list) else [n])]
        step = ast.FunctionDef("_meta_step", ast.arguments(posonlyargs=[], args=[],
                               kwonlyargs=[], kw_defaults=[], defaults=[]),
                               [ast.Global(sorted(set(main_names) | {"_meta_finished"})),
                                ast.For(ast.Name("_meta_once", ast.Store()),
                                        ast.Call(ast.Name("range", ast.Load()), [ast.Constant(1)], []),
                                        event_body, [])], [])
        exec(_compile([step], "<adaptive physical event>"), ns)

        # Cache original functions before compiling alternatives into this same
        # globals dictionary. Compilation does not run any event or decision.
        canonical = {k: v for k, v in ns.items() if callable(v)}
        self.decisions = {p: _functions(program[p], ns) for p in POLICIES}
        ns.update(canonical)
        self._decision_names = frozenset().union(*self.decisions.values())
        if self._decision_names & _RUNTIME_OVERRIDES:
            raise RuntimeError("a decision closure now contains a physical executor")
        self._canonical = {name: canonical[name] for name in self._decision_names if name in canonical}
        self.idle_functions = {}
        for mode, p in (("legacy", "full"), ("coordinated", "full_coordinated_idle")):
            pm = _main(ast.parse(program[p]))
            defs = [n for n in pm.body if isinstance(n, ast.FunctionDef)
                    and n.name in {"plan_idle_fleet", "start_idle_action"}]
            exec(_compile([_Promote(main_names).visit(n) for n in defs], "<adaptive idle>"), ns)
            self.idle_functions[mode] = {n.name: ns[n.name] for n in defs}
        ns.update(canonical)
        self._coordinated_request = ns["request_coordinated_idle_replan"]
        self._idle_begin = ns["begin_background_return"]
        base_main = _main(ast.parse(program["myopic_ab"]))
        base_begin = next(n for n in base_main.body if isinstance(n, ast.FunctionDef)
                          and n.name == "begin_background_return")
        base_begin.name = "_meta_base_return"
        exec(_compile([_Promote(main_names).visit(base_begin)], "<adaptive return>"), ns)
        self._base_begin = ns["_meta_base_return"]
        # Reuse the engine's planner tables (graph-sized coordinates and reserve
        # distances) for both modes instead of allocating them twice more.
        engine_planner = ns["idle_planner"]
        engine_planner.close()
        self._idle_config = replace(engine_planner.config, processes=1)
        legacy = copy.copy(engine_planner)
        legacy.__class__ = IdleFleetPlanner
        legacy.config, legacy._pool = self._idle_config, None
        coordinated = copy.copy(legacy)
        coordinated.__class__ = CoordinatedIdleFleetPlanner
        self.planners = {"legacy": legacy, "coordinated": coordinated}
        # A single posterior, updated under every policy. Controllers keep their
        # own observations/rate gates, but all reference the same physical robots.
        # Immutable node->cluster tables (>500,000 entries on big maps) are
        # shared right after each controller is built to bound peak memory.
        rates = {float(k): float(v) for k, v in prior_rates.items()}
        demand = GammaPoissonDemandModel(ns["graph"], rates,
                                         prior_concentration=self.settings["prior_concentration"])
        bank = {"mle": ns["reservation_policy"]}
        bank["mle"].templates = defaultdict(partial(deque, maxlen=128), bank["mle"].templates)
        bank["mle"].audit_path = None

        def share(controller):
            controller.demand.cluster_by_node = demand.cluster_by_node
            controller.demand.cluster_sizes = demand.cluster_sizes
            controller.cluster_summary = bank["mle"].cluster_summary
        share(bank["mle"])
        for key, path, style in (("spatial", spatial_model, "spatial_posterior"),
                                 ("paper", paper_model, "paper_moving_average")):
            bank[key] = OnlineAnticipatoryReservation(
                ns["graph"], ns["robots"], ReservationFCNN.load(path), rates,
                prior_concentration=self.settings["prior_concentration"],
                horizon_min=ns["scenario"].duration_minutes,
                charger_power_w=DEFAULT_CHARGING_POWER_W, reservation_style=style,
                lookback_min=self.settings["reservation_lookback_min"])
            share(bank[key])
        ns.update(_meta_controllers=bank, _meta_observed=[], _meta_demand=demand,
                  _meta_finished=False, _meta_policy="reactive_insertion")
        ns["_meta_observe"] = self._observe
        ns["_meta_refresh_arrival"] = self._refresh_arrival
        ns["plan_idle_fleet"] = self._plan_idle
        ns["request_coordinated_idle_replan"] = self._request_idle
        ns["begin_background_return"] = self._begin_background
        self.refs = {k: ns[k] for k in ("graph", "oracle", "charger_index", "router")}
        self.refs.update({"model." + k: bank[k].model for k in ("spatial", "paper")})
        self.refs.update({"demand.cluster_by_node": demand.cluster_by_node,
                          "demand.cluster_sizes": demand.cluster_sizes,
                          "cluster_summary": bank["mle"].cluster_summary})
        self._install("reactive_insertion")
        self.initial_snapshot = self.snapshot()

    def _observe(self, order, now):
        ns = self.ns
        ns["_meta_observed"].append(order)
        ns["_meta_demand"].observe(order.pickup_node, order.importance, now)
        for key, controller in ns["_meta_controllers"].items():
            if key == "mle":
                controller.observe(order.pickup_node, order.importance, now)
                controller.observe_order(order, now)
            else:
                controller.observe(order.pickup_node, order.importance, now)

    def _refresh_arrival(self, now):
        refresh = self.ns["refresh_reservations"]
        if self.policy in MLE_POLICIES:
            refresh(now, arrival=True)
        else:
            refresh(now)

    @property
    def policy(self):
        return self.ns["_meta_policy"]

    @property
    def now(self):
        return float(self.ns["now"])

    @property
    def loss(self):
        return float(self.ns["metrics"].weighted_wait_objective)

    def _install(self, policy):
        """Point the shared namespace at one policy's decision functions."""
        ns = self.ns
        ns.update(self._canonical)
        ns.update(self.decisions[policy])
        ns["_meta_policy"] = policy
        key = controller_key(policy)
        ns["reservation_policy"] = ns["_meta_controllers"].get(key)
        if key is None:
            ns["refresh_reservations"] = _no_refresh
        ns["idle_demand"] = ns["_meta_demand"]
        for planner in self.planners.values():
            planner.demand = ns["_meta_demand"]
        ns["idle_planner"] = self.planners[self.idle_mode or "legacy"]

    def set_policy(self, policy, *, refresh=True):
        """Switch policies at a decision boundary.

        Re-selecting the active policy is a no-op, so "stay" is exactly the
        uninterrupted fixed policy (no extra reservation refresh or cache reset).
        """
        if policy not in POLICIES:
            raise ValueError(f"unknown policy: {policy}")
        ns = self.ns
        previous_policy = ns["_meta_policy"]
        if policy == previous_policy:
            return
        mle = ns["_meta_controllers"]["mle"]
        if previous_policy in MLE_POLICIES:
            mle._finish(self.now, None, "meta_policy_inactive")
            # Stored on the controller, so it is part of every snapshot.
            mle.meta_reentry_pending = True
        self._install(policy)
        ns["reservation_assignment"] = None
        ns["baseline_projection_cache"].clear()
        # Queue-aware route re-evaluation reads this forecast without a None
        # guard, also from physical executors. Rebuild it from the current state.
        ns["assignment_queue_forecast"] = ns["build_assignment_queue_forecast"](self.now)
        key = controller_key(policy)
        if key == "mle" and getattr(mle, "meta_reentry_pending", False):
            # Reentering MLE must evaluate the current physical state, not reuse
            # an assignment cached before another policy controlled the fleet.
            # A first entry keeps the original update schedule exactly.
            mle.next_update = min(mle.next_update, self.now)
            mle.meta_reentry_pending = False
        if refresh and key is not None:
            ns["refresh_reservations"](self.now)

    @property
    def idle_mode(self):
        if self.policy in {"full_coordinated_idle", "full_mle_reservation"}:
            return "coordinated"
        if self.policy in {"full", "full_no_nn", "full_queue_aware", "queue_mle_reservation"}:
            return "legacy"
        return None

    def _plan_idle(self, now, only_ids=None):
        if self.idle_mode is None:
            return
        self.ns["start_idle_action"] = self.idle_functions[self.idle_mode]["start_idle_action"]
        self.idle_functions[self.idle_mode]["plan_idle_fleet"](now, only_ids)

    def _request_idle(self, now):
        if self.idle_mode == "coordinated":
            self._coordinated_request(now)

    def _begin_background(self, robot, now):
        if self.idle_mode:
            return self._idle_begin(robot, now)
        self._base_begin(robot, now)
        if robot.is_moving and not self.ns["schedules"][robot.spec.id].orders:
            path = (int(robot.decision_node), *map(int, robot.remaining_route))
            distance = self.ns["path_distance"](path)
            decision = self.ns["decision_snapshot"](robot, now)
            self.ns["idle_intents"][robot.spec.id] = {
                "robot_id": robot.spec.id, "kind": "charge", "target_node": path[-1],
                "arrival_min": decision.time_min + distance / robot.spec.speed_mps / 60,
                "battery_arrival_wh": max(0., decision.battery_wh - distance * robot.spec.energy_per_meter_wh),
            }
            self.ns["build_charging_calendar"](now)

    def advance(self, until):
        if not math.isfinite(until) or until < self.now:
            raise ValueError("advance requires a finite nondecreasing simulated time")
        ns = self.ns
        while ns["events"] and ns["events"][0][0] <= until and not ns["_meta_finished"]:
            ns["_meta_step"]()
        ns["now"] = float(until)

    def outstanding(self):
        """True while any released or future order is still undelivered."""
        ns = self.ns
        return bool(ns["pending"] or ns["service_lock"]
                    or any(s.orders for s in ns["schedules"].values())
                    or ns["delivered"] != len(ns["scenario"].orders))

    def drain(self, *, event_limit=10_000_000):
        """Finish every retained order; never silently score a failed drain."""
        ns = self.ns
        count = 0
        while ns["events"] and not ns["_meta_finished"]:
            ns["_meta_step"]()
            count += 1
            if count >= event_limit:
                raise RuntimeError("drain event safety limit reached; label is invalid")
        if self.outstanding():
            raise RuntimeError("incomplete rollout: do not use as a low-loss label")
        return self.loss

    def cutoff_arrivals(self, horizon):
        """Keep a common workload through H; preserve all existing commitments."""
        ns = self.ns
        ns["events"] = [e for e in ns["events"] if e[3] != "order_arrival" or e[0] <= horizon]
        heapq.heapify(ns["events"])
        ns["scenario"] = replace(ns["scenario"], orders=tuple(
            o for o in ns["scenario"].orders if o.request_time_min <= horizon))

    def snapshot(self):
        data = {k: self.ns[k] for k in STATE_NAMES}
        stream = io.BytesIO()
        _Writer(stream, self.refs).dump({"version": SNAPSHOT_VERSION, "source": self.source_digest,
                                       "inputs": self.input_digest, "state": data})
        return stream.getvalue()

    def restore(self, blob):
        data = _Reader(io.BytesIO(blob), self.refs).load()
        if (data["version"] != SNAPSHOT_VERSION or data["source"] != self.source_digest
                or data["inputs"] != self.input_digest):
            raise ValueError("incompatible simulator snapshot")
        self.ns.update(data["state"])
        for name in ("pair_distance_cache", "coord_cache", "baseline_projection_cache"):
            self.ns[name].clear()
        self._install(self.policy)

    def state_digest(self):
        """Hash of everything the remaining simulation can depend on.

        Equal digests at the start of a scoring tail imply identical tails
        (the simulator is deterministic), so the tail loss can be reused.
        Unequal digests only cost a recomputation.
        """
        ns = self.ns
        data = {k: ns[k] for k in STATE_NAMES if k not in DIGEST_EXCLUDED}
        key = controller_key(self.policy)
        data["_controller"] = ns["_meta_controllers"][key] if key else None
        stream = io.BytesIO()
        _Writer(stream, self.refs).dump(data)
        return hashlib.sha256(stream.getvalue()).hexdigest()

    def close(self):
        for planner in self.planners.values():
            planner.close()
        self.ns["_meta_controllers"]["mle"].scorer.close()
        if sys.modules.get(MODULE) is not None and sys.modules[MODULE].__dict__ is self.ns:
            del sys.modules[MODULE]
