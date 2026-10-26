# New switching-model training recovery — 26 October 2026

The human authorized continuing the failed new-model campaign, removing time-based fit aborts, and dedicating safe available VM compute to training. No final-test launch was authorized.

## Failure

The original development supervisor launched on 25 October at 20:45 UTC and stopped at 21:48:20 UTC (23:48 Israel). No person requested that stop. The controller received a successful saved XGBoost model pointer, waited only one second for process cleanup, then raised `Fit failed` if the child was still alive. The failure is not evidence of a failed statistical fit. An analogous one-second condition existed for completed episode workers.

## Compatible execution-only fix

`execution_recovery.py` retains a successful terminal result between nonblocking polls, waits for child exit without an arbitrary success-exit deadline, and still rejects genuine error messages, nonzero exit codes and missing results. Fit wall-clock aborts are disabled. Memory safeguards and owned-process failure cleanup remain.

The original sealed `src/fleet_switching` sources were not edited. `resume_training.py` loads a transparent, separately hashed execution override. This preserves the original campaign, science and learner identities and avoids invalidating compatible labels/models. No dispatch, projection, teacher, feature, optimizer or scenario logic changed.

- Original execution file SHA-256: `d0cbd9dd2bba0665cc01f46f592981877abfcc64a37960af49024fe38aec6f90`
- Execution override SHA-256: `20cd1498fce964794fd293747c0773019be1f776b14d217ec7386e9ce5e2b956`
- Campaign ID: `a54865fbefbd86217a12b59c4eac2dc2d8153d66617bd3c5fcb7d04dae0eb68e`
- Science ID: `83b76cbdab4d5e9ad8bfc7ccd8052f8a9b64d4b96e9c555427e7c9dfb713318b`

Thirteen focused process-control tests passed on the VM before launch. Tests cover delayed successful fit/episode cleanup, disabled elapsed fit abort, actual errors/nonzero exits, missing results and the memory guard. They do not retrain a test model or run a production simulation.

## Preserved work and resource priority

615 completed training episode receipts existed before recovery, including two that finished during original shutdown. 4,993 existing model/input/checkpoint/receipt artifacts were hashed and verified before resume. The replay database and original status/identity documents were backed up under `recovery/20261026/before_resume`.

All remaining old comparison CPU work was paused, not terminated:

- `robot-static-policy-models-20261024.service` frozen, preserving its two Beijing workers.
- `robot-static-policy-approved-resume-20261024.service` frozen so the watchdog cannot resume old work.
- Original Beijing 000 coordinated worker and exactly five verified descendants received SIGSTOP, preserving memory.
- The two original full-campaign supervisors remain frozen; all other original stopped workers remain stopped.

Exact identities and artifact hashes are recorded in `comparison_paused.json`. Do not automatically thaw these services or workers after training. Their latest static-selector summary is 27/48; it is preserved unfinished work.

Execution admission now permits up to 67 episode processes plus one four-thread fitter on 72 logical CPUs, reserving one logical CPU for coordination. Actual concurrency remains limited by the existing 64-GiB available-RAM reserve, 8-GiB trainer allowance and prospective 1.5-GiB worker allocation. The first resumed collection sample had 66 active episode workers and about 138 GiB available RAM. No arbitrary training/fit/service runtime cutoff is imposed.

## Restart and scope

Detached service: `robot-switching-training-recovery-20261026.service`

Launch: 26 October 2026 at 07:16:32 UTC (09:16 Israel).

Initial supervisor PID 1841893; child PID 1841895, start ticks 109909569. Verify current identity from `RECOVERY_LAUNCH.json` and `/proc` before any process action; do not trust a PID alone.

Readable entry: `recovery/20261026/resume_training.py --execute`. Never run a second copy. The supervisor and campaign locks prevent duplicates. Existing completed episodes are reused and unfinished episodes resume committed boundaries. Both XGBoost and NN keep their own trajectories and saved replay.

Scope remains 1,600 training actor episodes and 2,400 matched development validation outcomes (two selectors plus ten fixed policies across 200 development inputs). Final cities Cambridge UK, Ghent, Salamanca and Lucca remain unopened. There is no automatic final-test step.

## Paths and monitoring

VM campaign root:

`/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_small_city_switching_v4_pilot_fixes_20261025`

- `full_run_status.json`: current supervisor stage and total counts.
- `training_status.json`: active collection, replay and fit progress.
- `recovery/20261026/training_resume.log`: new controller output, separate from original failure logs.
- `recovery/20261026/execution_override_active.json`: unchanged identities plus execution settings.
- `recovery/20261026/RECOVERY_LAUNCH.json`: exact command and initial process identity.
- `recovery/20261026/before_resume/`: preserved pre-recovery status/replay backup.
- `recovery/20261026/recovery_failed.json`: only if this recovery fails.

The previous static-comparison monitor was redirected to this training recovery, preserving its existing cadence and quiet-normal intent. It must report genuine failures promptly and keep old work paused.

Provisional estimate from recovery: roughly 2–4 additional hours for collection/final fitting and 4–8 hours including the development comparison. These are planning ranges, not confidence bounds; update from measured resumed progress and hard-city tails, and keep final-test time separate.

## Automatic recovery enabled without interrupting healthy training

At the user's further request on 26 October, `enable_automatic_recovery.py` installed a readable systemd user-service drop-in. The service will restart after failure with a 30-second delay and no runtime limit. Its future entry is `resume_training_v2.py --execute`, which preserves immutable inputs/completed outcomes while allowing legitimately advancing checkpoints and current-model pointers. Previous launch receipts are archived. The healthy supervisor PID remained unchanged; no running episode was restarted to apply this setting.

Fifteen focused regression tests passed. `automatic_recovery_enabled.json` records the before/after service settings. Normal completion does not restart the campaign, and an explicit systemctl stop remains an explicit stop. The monitor now checks every five minutes, stays quiet while healthy, and investigates repeated restarts or loss of progress instead of leaving the service idle overnight. Integrity failures are not permission to bypass scientific compatibility checks. Old benchmark simulations remain paused.
