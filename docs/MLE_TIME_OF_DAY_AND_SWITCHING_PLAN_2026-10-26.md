# Full MLE: time-of-day diagnosis

Read-only training-label and development-controller analysis. No final-test outcome was used and no new simulation or fitting was run.

## Training target by simulation hour

Rows use the start of each 15-minute decision window. Drain after hour 12 is separate. Best includes ties within 1e-7 raw loss units.

| Simulation interval | Decisions | Best/tied % | Unique best % | All ten tied % | Best/tied %, excluding all-ten ties | Mean Full MLE regret / robot |
|---|---:|---:|---:|---:|---:|---:|
| 0–1 h | 6,400 | 92.48 | 0.98 | 58.78 | 81.77 | 0.00747 |
| 1–2 h | 6,400 | 87.77 | 2.17 | 45.92 | 77.38 | 0.01467 |
| 2–3 h | 6,400 | 85.84 | 3.30 | 40.34 | 76.27 | 0.02391 |
| 3–4 h | 6,400 | 83.89 | 4.38 | 35.88 | 74.88 | 0.02952 |
| 4–5 h | 6,400 | 79.03 | 6.42 | 27.83 | 70.95 | 0.05530 |
| 5–6 h | 6,400 | 77.84 | 7.89 | 23.38 | 71.08 | 0.08884 |
| 6–7 h | 6,400 | 77.95 | 8.61 | 23.97 | 71.00 | 0.11017 |
| 7–8 h | 6,400 | 80.58 | 6.84 | 26.88 | 73.44 | 0.08157 |
| 8–9 h | 6,400 | 81.30 | 7.56 | 26.91 | 74.41 | 0.06467 |
| 9–10 h | 6,400 | 84.03 | 7.41 | 28.73 | 77.59 | 0.05942 |
| 10–11 h | 6,400 | 84.69 | 6.47 | 32.59 | 77.28 | 0.05565 |
| 11–12 h | 6,380 | 86.32 | 6.25 | 35.28 | 78.86 | 0.02898 |
| After 12 h (drain) | 4,815 | 99.85 | 0.00 | 99.21 | 81.58 | 0.00119 |

## MLE activation audit

Across all 200 fixed-Full-MLE development validation episodes: **0 episodes ever activated MLE reservations**, 0 activations, and 0 candidates scored.

The exact cumulative controller counters were decoded from terminal checkpoints without constructing or advancing a runtime. Saved decision features give a separate sampled-state cross-check.

Rate counts and request templates reset on fixed 60-minute epochs. Arrivals are observed under every active benchmark, including while MLE is inactive. The epoch confidence error allocation also becomes stricter with epoch number. Thus later clock time does not mean a larger cumulative MLE rate sample.

Reason counters:
- arrival_horizon_ended: 192
- insufficient_arrivals: 9222
- rate_intervals_too_wide: 376

## Interpretation and possible cutoff experiment

A prefix policy can collect arrivals before switching because the controller is already observed in the background. However, choosing a cutoff must use development data and complete episode loss, including drain. The original final-test results must remain a report of the original frozen experiment.

Saved counterfactual costs cannot simply be summed or spliced to evaluate a new schedule: switching changes routes, batteries, queues and later states. A genuine cutoff comparison needs fresh full-episode runs in a separate development output, preserving commitments at the switch.

The JSON report includes per-family, city, profile, three-hour and within-hour phase breakdowns plus per-policy cost differences. Any time trend here is descriptive and cannot establish that MLE accuracy caused it.

## Assessment

The recorded labels do not show a monotonic later-is-better pattern. Excluding all-ten-policy ties, Full MLE is best/tied in 78.12%, 72.18%, 72.93% and 77.89% of decisions in hours 0–3, 3–6, 6–9 and 9–12 respectively. The ordinary-demand profile shows the same broad pattern (78.84%, 73.31%, 74.36%, 76.02%); the result is not solely the explicit surge profile.

The stronger finding is that the reservation gate never opened in any of the 200 fixed-Full-MLE development validation episodes. There were 9,222 insufficient-arrival planning outcomes and 376 overly-wide-rate-interval outcomes, with zero reservation activations and zero alpha candidates scored. This is observed behavior of the current policy, not proof that the dispatcher/idle parts are inactive. Full MLE combines multiple components, and its benchmark wins alone cannot be attributed to MLE reservation learning.

The current controller uses a 60-minute fixed rate epoch, clears request templates at an epoch change, and observes every released request even when another benchmark controls the fleet. A warm prefix therefore does preserve current-epoch history, but waiting several hours does not create a pooled multi-hour MLE rate sample. Whole-hour cutoffs also coincide with a rate reset.

## Concrete follow-up design

1. Diagnose whether the minimum-arrival and uncertainty gates are appropriate for small-city traffic. Do not simply loosen statistical thresholds until reservations activate; retain the original policy/results as the reference.
2. In a **separate development-only output**, compare prefixes Reactive Insertion, Full without NN and Full Coordinated Idle, followed once by Full MLE. Candidate cutoffs in hours: **0, 0.25, 0.5, 0.75, 1, 2, 3, 4, 6**, plus each pure-prefix baseline. The zero cutoff is the existing always-Full-MLE reference and needs no duplicate run per prefix.
3. Screen a balanced subset of training scenarios, refine promising cutoffs, then select on the disjoint development validation inputs. Preserve released-order observations, current edges, charging queues, handling commitments and the 12-hour arrival stream; drain every admitted order and compare final episode loss.
4. Do not select a cutoff by summing counterfactual row costs from the existing actor trajectories. A new schedule changes the states it visits. The completed final-test report remains the original frozen experiment; a revised policy needs separate confirmatory evaluation and must not be tuned on those test outcomes.

No cutoff sweep has been launched. This turn performed read-only analysis of existing training and development artifacts only. There is no supported preferred cutoff yet.
