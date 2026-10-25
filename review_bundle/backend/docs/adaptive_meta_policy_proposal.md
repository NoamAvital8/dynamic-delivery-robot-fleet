# Adaptive Meta-Policy for Dynamic Robot Delivery

## Research objective

**Question:** Can an adaptive algorithm-selection policy outperform the best single fixed dispatching policy across different cities, workloads, and changing demand regimes?

The target is the project's **total importance-weighted delivery loss**, including deadline penalties, not delivery time alone.

## Recommended design: a hybrid adaptive selector

Train a model offline to predict how each candidate dispatching policy will perform under the current operating conditions. During online operation, periodically select the policy with the lowest predicted loss, but switch only if the expected gain is sufficiently large.

Let the candidate policies be \(\Pi=\{\pi_1,\ldots,\pi_K\}\) and the observable fleet state be \(s_t\). Train estimates of their future loss over horizon \(H\):

\[
\hat L_i(s_t)\approx\mathbb E[L_H\mid s_t,\pi_i].
\]

Select:

\[
\pi^*(s_t)=\arg\min_{\pi_i\in\Pi}\hat L_i(s_t).
\]

**Predict loss per policy rather than merely a winning policy label.** A predicted difference of 100 versus 101 warrants less confidence in switching than 100 versus 160.

## Why not select one algorithm per city?

A city-specific fixed choice overlooks changes in demand peaks, the mix of urgent jobs, robot availability, battery reserves, and congestion. City identity can be a feature, but the model should primarily respond to **the current operating regime**. Train across cities and test generalization to an unseen city (leave-one-city-out testing).

## Input features

Use compact features available *at the decision time*, not future realizations:

- Recent order arrival rate and trend; pending orders per available robot.
- Busy and charging robot fractions; mean/minimum battery levels.
- Mean deadline slack, share of urgent orders, and recent lateness.
- Spatial pickup concentration/entropy and mean pickup-to-drop-off distance.
- Robot speed/capacity heterogeneity and travel-time uncertainty.
- Recent importance-weighted loss.

Initially compare simple tabular models (decision tree, random forest, gradient boosting) against a small neural network. Use a **joint model across cities**, not one network per city by default.

## The second idea: run competing algorithms on the first n orders

There is a state-consistency problem: after the same orders, different algorithms leave robots at different locations, with different batteries, plans, pending requests, and delivery outcomes. You cannot freely transplant the winning algorithm's simulated state into the real system.

**Improved approach: cloned-state rollout.** At each reevaluation time, clone the *actual current* fleet state \(S_t\) and simulate each candidate from that identical starting state for the next \(H\) orders or minutes:

\[
S_t\xrightarrow{\pi_1}L_H^1,\qquad
S_t\xrightarrow{\pi_2}L_H^2,\qquad\ldots\qquad
S_t\xrightarrow{\pi_K}L_H^K.
\]

Compare future incremental losses, accounting consistently for jobs unfinished at the rollout horizon. In an online system, future orders are **unknown**: sample plausible futures from an estimated demand model, and evaluate each candidate on matched samples. Running the *actually realized* future is valid only for a clearly labeled **offline clairvoyant oracle**, not a feasible online selector.

This simulation-based **policy tournament / rollout selector** is computationally expensive. It is useful both as a benchmark and as an offline training-label generator for the learned selector.

## Should selection happen once or repeatedly?

**Repeatedly, but not every order.** Use periodic reevaluation with a minimum policy holding period and a switching threshold:

\[
\hat L_{\mathrm{current}}(s_t)-\hat L_{\mathrm{candidate}}(s_t)>\delta.
\]

Here \(\delta\) is a penalty or required gain that discourages oscillation. Alternatively optimize predicted loss plus explicit switching cost.

Illustrative parameters **to tune, not assume correct**: hold for at least 20 orders; thereafter reconsider every 10 orders; switch only if predicted improvement is at least 5%. Compare these with one-time selection and event-triggered reevaluation.

**Safety constraint:** Changing policy affects future choices only. It must not reset positions, battery, active routes, already picked-up cargo, or binding commitments. All policies should work with a shared canonical simulator state.

## Training procedure

1. Generate varied training scenarios across cities/graph structures, demand intensities, spatial distributions, fleet heterogeneity, and uncertainty levels.
2. Collect fleet-state snapshots at many points across scenarios, preferably using **multiple behavior policies** to capture diverse states.
3. From each snapshot, clone the same state and evaluate all candidates using matched simulated future demand/travel-time realizations.
4. Record future incremental importance-weighted loss for each candidate, with consistent treatment of unfinished orders and computational overhead.
5. Train on examples \((x_t,L_1,\ldots,L_K)\), with only features observable at time \(t\).
6. Tune models, rollout horizon, decision frequency, and switching threshold on **validation scenarios**.
7. Evaluate the *actual online meta-policy* on held-out test scenarios. Report delivered results, not only model prediction accuracy.

## Comparison methods

| Method | Definition | Role |
|---|---|---|
| Best fixed policy | Choose a single policy based on training/validation; use it for the entire test | Main strong baseline |
| Static selector | Choose a policy once, at the start of a city/day/scenario | Tests whether adaptation matters |
| Online rollout selector | Periodically clone state and simulate all candidates using estimated futures | Computationally intensive benchmark |
| **Learned adaptive selector** | Periodically predict losses and select subject to switching rules | **Proposed method** |
| Clairvoyant oracle | Choose using future information unavailable at decision time | Optimistic hindsight reference only |

Compare against **every** candidate dispatching policy, not just their average. A per-scenario best-fixed hindsight oracle and an oracle capable of changing policies with future knowledge are distinct upper-reference concepts and must not be conflated.

## Candidate policies from the current project

The project handoff identifies six benchmark algorithms:

| ID | Policy |
|---|---|
| `full` | Full posterior/NN reservation plus predictive idle actions |
| `myopic_ab` | Battery-feasible earliest-completion myopic method |
| `reactive_insertion` | Reactive exact incremental-loss insertion |
| `paper_sa_adapted` | Adapted Ghiani et al. anticipatory reservation baseline |
| `full_no_nn` | Full policy without NN reservation |
| `full_no_idle` | Full policy without predictive idle control |

A smaller, diverse subset can be used for the first prototype. The paper-adapted baseline is **not** an exact reproduction of the original paper's experiments.

## Evaluation design

**Primary metric:** total importance-weighted delivery loss. **Secondary metrics:** on-time/late/undelivered orders, completion rate, mean and tail delay, energy, robot utilization, number of switches, selection overhead, and total compute time.

Suggested ablations:

- One-time versus periodic selection.
- Learned predictor versus online rollout.
- With versus without hysteresis.
- Short versus long rollout horizon \(H\).
- Different reevaluation intervals \(m\).
- NN versus simpler prediction models.

Use matched test scenarios, multiple seeds, paired comparisons and uncertainty intervals. Ensure selection time fits operational deadlines. Test shifts **within** cities and leave-one-city-out transfer.

The handoff describes a five-city campaign (Tel Aviv, Haifa, Manhattan, New York City, Barcelona), with 20 training and 5 held-out test scenarios per city and six policies. Its results are a useful foundation, but the existing campaign alone does **not** necessarily include the shared-state counterfactual rollouts needed for meta-selector training. Do not assume the campaign is complete without checking its status, and do not overwrite its outputs.

## Research novelty and connection to papers

Ghiani, Manni & Manni (2022), *A scalable anticipatory policy for the dynamic pickup and delivery problem*, use online instance features and supervised learning to tune **parameters of a base policy**. Here the model would choose **among different dispatching algorithms**, possibly repeatedly, based on predicted loss and switching cost. This is a research direction, not a claim that such algorithm selection is entirely new in the literature.

The project's original proposal also calls for evaluating algorithms across fleet heterogeneity, workload, uncertainty, and anticipatory planning. Adaptive selection extends that empirical question: *under what conditions is each solver worth using, and does adaptation beat the best fixed solver?*

## Suggested implementation order

1. Standardize all candidate policies to operate on a common, legally valid simulator state.
2. Establish and verify individual fixed-policy results.
3. Implement cloned-state rollout for offline counterfactual evaluation.
4. Generate training-only state-to-policy-loss examples.
5. Train tabular regressors, then compare with an NN.
6. Add periodic selection and switching hysteresis.
7. Run held-out multicity evaluations and test generalization to unseen cities.

**Final recommendation:** build a **periodically updated, learned per-policy loss predictor with a switching threshold**. Treat cloned-state rollout as the principled computational benchmark and training tool, and once-only/per-city choices as baselines.

## Source material

- Ghiani, G., Manni, A., & Manni, E. (2022). *A scalable anticipatory policy for the dynamic pickup and delivery problem*. Computers & Operations Research 147, 105943 (uploaded PDF).
- Jeong, J., & Moon, I. (2024). *Dynamic pickup and delivery problem for autonomous delivery robots in an airport terminal*. Computers & Industrial Engineering 196, 110476 (uploaded PDF).
- Project proposal: `הצעת_פרויקט_רובוטי_משלוחים.pdf`.
- Project handoff: `ICAPS_CAMPAIGN_HANDOFF.md` (timestamped campaign information).

## Agreed data separation — 15 October 2026

Generate new, independent scenarios for all meta-selector training and
validation. Do not reuse any scenarios already used by the fixed-policy
campaigns for these purposes. Keep their exact original scenarios, maps, seeds
and configured priors for a matched comparison of the frozen meta-selector
against saved baseline results. Reject reused seeds and duplicated workloads,
including renamed/reseeded copies. The implementation guide contains the fresh
data generator and separate training/comparison manifest commands. The existing
benchmark comparison is distinct from an additional untouched final test set.
