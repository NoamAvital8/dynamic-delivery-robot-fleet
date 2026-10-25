# Adaptive meta-policy: changes after the 15 Oct 2026 review

Branch `meta-policy-fixes` (worktree `dynamic-delivery-robot-fleet-meta-policy`),
based on `codex/new-city-benchmark-suite` @ `735942f` plus the reviewed,
previously uncommitted meta files. No training, benchmark or campaign was run;
only tests. IDs refer to the review report (P = priority finding, O =
optimization, A = item added from the second review).

## Correctness

| ID | Change | Where |
|---|---|---|
| P0-2 | MLE policies keep the original `refresh_reservations(now, arrival=True)` semantics: a new arrival is not counted as an existing backlog, so an active MLE reservation is no longer cancelled on every arrival. | `runtime.py` `_refresh_arrival` |
| new | Entering an MLE policy for the first time keeps its original update schedule; only a re-entry after the selector switched away forces an immediate re-plan (flag stored on the controller, so it is snapshotted). Before, entering at t=0 made an extra MLE planning call. | `runtime.py` `set_policy` |
| P1-8 | `reservation_assignment` is part of the snapshot; re-selecting the active policy is a no-op (no extra reservation refresh, no cache reset), so "stay" equals the uninterrupted fixed policy. | `runtime.py` |
| new | The queue-aware `evaluate_sequence` reads the global queue forecast without a `None` guard, also when physical route execution re-evaluates a route. The old adapter set it to `None` on every switch and restore. It is now snapshotted and rebuilt from the current state on every switch. | `runtime.py` |
| P0-4 | `decide_until: all_delivered` keeps making decisions after the last arrival until every order is delivered (default for new manifests). The old behaviour is `arrival_horizon`. | `pipeline.py` |
| A6 | Adaptive holding: a longer hold now needs the same winner, with a margin, at every shorter tested horizon. | `learning.py` `choose` |
| P1-2 | The first decision of an episode has no incumbent, so no hysteresis applies. | `learning.py`, `pipeline.py` |
| P3 | `rate_trend` is a real rate difference and is 0 until two windows exist; `map.cluster_noise_share` (always 0) removed; feature extraction no longer writes simulator caches. Feature `VERSION` = 2. | `features.py` |
| P3 | The decision-function closure is re-applied over the canonical functions at every switch; a closure containing a physical executor fails closed. | `runtime.py` `_install` |

Corrections to the review:
- `assignment_queue_forecast` is not harmless (see the new row above).
- Near the end of the arrival period labels are not artificially small: after 720 min there are simply no further arrivals to admit.
- Draining a backlog is cheap. The runner logs show that nearly all simulator time is spent while orders arrive, and the post-arrival phase is almost free. So O6 (a terminal value instead of a drain) has little value; the real costs are the arrival phase, slow candidates (`reactive_insertion`) and the giant maps.

Verified: the actual episode continues only from the chosen policy's branch. Alternatives are scored and then discarded, and nothing expands into a tree. `test_actual_trajectory_follows_only_the_chosen_policy` replays the recorded choices without any alternatives and reproduces the collected episode exactly.

## Identity, resume and operations

| ID | Change |
|---|---|
| P0-3 | Labels depend on a **simulator identity**: the ten generated programs plus the parsed local import closure of `runtime.py`, `labels.py` and `exact_runtime.py`, together with models, controller settings, label horizon and continuation. Learning, orchestration and unrelated scripts are excluded. Features have their own identity. Label code moved to `meta_policy/labels.py` (`LABEL_VERSION` 2). |
| A2 | New `--mode validate`: run a selector online only on its own held-out validation episodes. Provenance now separates fitting and validation scenarios (`build_provenance`). `evaluate` still rejects everything that informed the selector. |
| P1-2 | `scripts/configure_meta_selector.py` writes selector variants (`--margin-scale`, `--minimum-gain-fraction`, `--adaptive-holding`) for online threshold tuning. |
| P0-1 | `--max-steps N` pilot mode (no drain, resumable later) and `scripts/summarize_meta_run.py` (rollouts, tail reuse, window/tail wall time, peak RSS, rough extrapolation). Receipts record window/tail time and peak RSS. |
| §6 | One failed episode no longer cancels or stalls the others; failures go to `failure.json` and are raised at the end. |
| A5 | `progress.json` is finalized (`complete` / `pilot_stopped` / `failed`); status files record PID, host, boot id and process start time. |
| O7 | Episodes start longest-first; each episode gets a fresh worker process (`max_tasks_per_child=1`); `close()` drops the engine module. |
| A3 | The exact index is SHA-256-verified once, then trusted while size/mtime/inode are unchanged (`meta_verified.json`). The cache folder can differ between simulators in one process, and `None` restores the in-process index. |
| A8 | Idle planners share the engine planner's graph-sized tables; controller cluster tables are shared right after each controller is built. |
| A9 | Every scientific controller setting (prior concentration, look-back, idle, queue and MLE settings) is validated, passed to the engine, recorded in manifests and part of the input and label identities; unknown keys fail. |
| A4 | The benchmark runner strictly validates results (seed, order count, completeness, finite non-negative metrics) on run, reuse and adopt paths, and stamps a result checksum that is re-checked on reuse. |
| §6 | Resolved selector path in the run signature; training defaults to 8 threads instead of 60. |

## Compute

| ID | Change |
|---|---|
| O1 | Default portfolio is six complementary policies (`full_no_idle`, `paper_sa_adapted`, `full_queue_aware`, `full_coordinated_idle`, `queue_mle_reservation`, `full_mle_reservation`), chosen from the 35 completed test scenarios of the 9 training cities. The four queue-aware policies give legacy and coordinated idle control each with and without NN reservation (MLE never activated in 72 runs). `reactive_insertion`, `myopic_ab`, `full` and `full_no_nn` stay selectable but are not defaults. Beijing and Sydney are excluded from training by default (`--exclude-cities`). |
| O2 | New manifests default to `full_no_idle` as the continuation policy (configurable). The ranking agreement with other continuations must be checked in the pilot (P1-6). |
| O3 | Exact deduplication of scoring tails: a digest of the tail-start state (bookkeeping counters excluded) lets identical tails be computed once. |
| O4 | `label_step_probability` (default 0.25 for new manifests): only sampled decisions get the full tournament; the others run the chosen policy only. |
| P1-3 | Sticky exploration (`stickiness` 0.75) and fixed-behaviour episodes (`fixed_episode_fraction` 0.2) for bootstrap rounds. |

## Learning

| ID | Change |
|---|---|
| P1-1 | Default target `advantage`: per-policy loss minus portfolio mean, plus a level head. `absolute` remains available. |
| P1-2 | Switching margins use episode-weighted q90 residuals of predicted pairwise differences, scaled by `margin_scale`. |
| A7 | Validation reports raw-argmin and deployed-rule regret, stay and best-fixed regret, top-1, Spearman, switch and veto rates, holding distribution — per row, per episode and per city. |
| A9 | The NN uses all rows with episode-weighted minibatch losses and weighted normalization (no bootstrap). |
| P1-4 | `--exclude-city` / `--validation-city` for leave-one-city-out fitting. |
| §5 | `prepare_meta_manifest.py` accepts several `--reference-suite` (e.g. 5 + 6 cities) and writes all recommended settings explicitly. |

## Tests

`tests/test_meta_fixes.py` (31 tests) covers:

- MLE parity with an active planning path, and an injected active assignment surviving a new arrival.
- Fixed-policy parity of six policies under a pending backlog with 2-minute boundaries and snapshot/restore.
- No-op re-selection, MLE re-entry, snapshot of the reservation assignment and queue forecast, and tail-digest soundness.
- Controller settings reaching the engine, and index verification being reused (Linux).
- Tail deduplication, sampled labels, sticky exploration, post-horizon decisions, pilot stop/resume, failure isolation, and validate/evaluate separation.
- Label-identity scope, read-only features, and the two second-review learning cases (rule regret, horizon agreement).
- Advantage margins, benchmark result validation, stamp checksums, and multi-suite preparation.

Existing tests were updated only for the stricter result validation and the new failure reporting.

## Still open (needs a pilot or a decision)

- **O5:** parallel candidate branches inside one episode (fork-based workers). This needs measured copy-on-write memory growth on the big maps.
- **O6:** a terminal value instead of a full drain. This is low priority: drains turned out to be cheap.
- **P1-6:** validate the continuation choice (rank agreement) and the 60-minute horizon in the pilot.
- Per-city memory classes for worker admission.
- The static per-city baseline, fixed-policy runs on fresh validation scenarios, and the paired comparison report.
- Recovering the stopped v7 campaign from its frozen checkout. That checkout still contains the three meta scripts, which change its benchmark digest. Move or remove them there before resuming (not done here).
