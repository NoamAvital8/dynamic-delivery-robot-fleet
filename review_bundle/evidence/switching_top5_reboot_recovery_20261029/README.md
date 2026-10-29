# Five-benchmark training recovery after VM reboot — 29 October 2026

The service was cleanly stopped by systemd at **14:34:41 Israel time** and
stopped at 14:34:43. The VM booted again at approximately **14:39**. The
transient user-service definition did not survive reboot, so training did
not restart automatically. The available journal does not identify who or
what initiated the reboot.

Saved progress was 1,245/1,600 completed training episodes (78%), with 32,277
committed replay states for each learner. Validation had not started.
Scientific campaign identity, all saved model hashes and SQLite `quick_check`
passed before resumption. No completed episodes, model weights or replay
were reset. Training resumed at **15:02:50 Israel time**.

The same service is now a persistent user unit enabled for `default.target`,
with user lingering already enabled. It retains 67 episode workers, four fit
threads, the memory reserve, 30-second failure restart and no runtime limit.
Conditions skip startup once development completes or final test is opened.
No scientific source files, hyperparameters, scenario manifests or candidate
policies were changed. Other VM jobs were not started, stopped or changed.

`POST_RESUME_STATUS.json` verifies resumed progress beyond the stopped counts,
an active enabled service, new learner updates and no test artifacts. The
recorded CPU sample used 92.37% of 72 logical CPUs with 67 workers. The initial
remaining-time estimate was **90–130 minutes including validation**, based on
remaining city-weighted training costs and the original matched validation
costs, with allowances for fitting and the slow final episodes.

- `PRE_RESUME_VERIFICATION.json`: saved campaign/model/replay verification.
- `RESUMED.json`: persistent unit definition, enabled state and resume time.
- `POST_RESUME_STATUS.json`: dated progress and holdout check; not live status.
- `DIAGNOSTIC_JOURNAL.txt`: stop, reboot boundary and resumed service evidence.
- `robot-switching-top5-no-full-mle-20261029.service`: installed user unit.
- Recovery helper: `operations/resume_switching_top5_after_reboot.py`.

The persistent unit resides in the VM user's `.config/systemd/user/`. The
original sealed 60-file build manifest remains unchanged, as do the saved
scientific identities and both original ten-action model artifacts.
