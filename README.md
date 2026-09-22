# Dynamic Delivery Robot Fleet

Course project for **AI and Autonomous Systems**.

We study online planning and assignment for a heterogeneous fleet of autonomous delivery robots operating on a city graph under stochastic demand and travel times.

## Core setting

- Pickup-to-delivery requests arrive online.
- Each request has a pickup, destination, package size/weight, importance level, request time, and deadline.
- A larger numerical importance value means a more important request.
- Robots are heterogeneous and may differ in speed, payload/volume capacity, battery capacity, and energy consumption.
- Robots may already have assigned packages, so a new request may be inserted into an existing route.
- The planner may wait for a better robot instead of assigning immediately.
- Future extensions include package transfer between robots and proactive repositioning.

## Primary objective

The current benchmarks use a deadline- and importance-aware delivery loss. For a request with request-to-delivery time \(T\), allowed duration \(D\), and importance \(w\):

$$
L(T,D,w)=w\min(D,T)+(w+1)^2\max(0,T-D).
$$

The first term charges regular importance-weighted loss before the deadline, while lateness receives a much larger importance-dependent penalty. Policies minimize this loss; they do not minimize delivery time by itself.

## Planned benchmark policies

1. Nearest-Available Robot (NAR)
2. Myopic A — earliest feasible completion
3. Myopic B — availability + battery aware
4. Reactive Insertion (RI)
5. Reassignment Policy (RP)
6. SA-adapted / anticipatory policy
7. Proposed algorithm
8. Optional offline oracle for small instances

# Proposed anticipatory algorithm

The proposed method extends the capacity-reservation idea of Ghiani et al. to a **heterogeneous robot fleet**, and adds spatial demand estimation and computationally cheap robot pruning.

At a high level:

**online demand estimation → FCNN reservation fractions → choose concrete reserved robots → cheap candidate scoring → top-\(K\) shortlist → exact greedy loss minimization**

## 1. Heterogeneous FCNN reservation fractions

Ghiani et al. use a multi-layer feed-forward neural network to map estimated demand features to the capacity-reservation vector \(\alpha\).

In the original paper:

- the input has \(C+1\) values;
- the first \(C\) inputs are the predicted proportions of requests in each priority class;
- the last input is the predicted total number of requests per vehicle;
- there is one hidden layer with \(C\) neurons;
- the activation function is \(\tanh\);
- the output is \(\alpha=(\alpha_1,\ldots,\alpha_C)\);
- \(\sum_c \alpha_c=1\);
- training targets are generated offline by testing candidate \(\alpha\) configurations under perfect information and selecting the best one.

Our extension is to a heterogeneous fleet. For robot type \(k\) and priority class \(c\),

$$
\alpha_{k,c}
$$

denotes the fraction of robots of type \(k\) assigned to the reservation stratum that may serve class \(c\) and higher-priority classes.

The implemented model keeps the paper's **\(C+1\) inputs**:

1. the predicted whole-horizon share of each of the \(C\) importance classes;
2. the predicted total number of requests divided by the fleet size.

At runtime, the paper's moving-average prediction is replaced by our size-aware Gamma-Poisson estimate. For importance \(c\), the predicted whole-horizon count is the number already observed plus the posterior mean arrival rate multiplied by the remaining horizon.

The network also keeps the paper's single hidden layer with \(C\) `tanh` neurons by default. Our heterogeneous extension changes the output from \(C\) values to one joint \(K \times C\) matrix. A separate softmax is applied to every robot type, so each type's fractions are non-negative and sum exactly to one. The hidden size remains configurable for ablation experiments.

`ReservationFCNN` is trained jointly with fractional cross-entropy targets. Following the paper, every training scenario is treated with perfect information **offline**: the real simulator is run with every configured fixed \(\alpha\) candidate, and the candidate having minimum final loss becomes that scenario's label. Test scenarios must never appear in this target-generation manifest.

`scripts/generate_reservation_training_data.py` evaluates scenario/candidate pairs with a process pool controlled by `--processes`. `scripts/train_reservation_fcnn.py` trains multiple independent initializations concurrently, also controlled by `--processes`, selects the seed with the smallest held-out validation loss, and refits that initialization on all training instances before saving it.

## 2. Spatial demand regions: HDBSCAN with full graph coverage

We will use **HDBSCAN** to discover spatial demand regions because it can identify clusters with different densities and does not require a single global \(\varepsilon\) value as DBSCAN does.

Raw HDBSCAN is not sufficient by itself because it may label some observations as noise and therefore does not guarantee that every graph node belongs to a cluster.

The implemented solution is:

1. run HDBSCAN to identify the high-density spatial structure;
2. choose a representative node for every resulting cluster;
3. assign every HDBSCAN noise node to its nearest cluster representative by graph shortest-path distance;
4. save the final cluster id directly on every graph node as `in_cluster`.

The clustering and the graph-distance coverage step are run **offline during graph initialization**, not in the online decision loop. The generated GraphML therefore already contains a complete partition and cluster lookup during simulation is \(O(1)\).

Thus HDBSCAN determines the demand structure, while the coverage step guarantees that every graph node belongs to exactly one cluster.

If this approach is unstable, graph \(k\)-medoids or another full-partition method will be used as a comparison baseline.

## 3. Bayesian arrival-rate estimate for each cluster and priority

For each spatial cluster \(z\) and priority class \(c\), requests are modeled as a Poisson arrival process with an unknown rate

$$
\lambda_{z,c}.
$$

We use a Gamma prior with the **rate** parameterization:

$$
\lambda_{z,c}\sim \mathrm{Gamma}(a_{z,c},b_{z,c}),
$$

so that

$$
E[\lambda_{z,c}] = \frac{a_{z,c}}{b_{z,c}}.
$$

After observing \(n_{z,c}\) requests during an observation time \(\Delta t\),

$$
\lambda_{z,c}\mid data
\sim
\mathrm{Gamma}
\left(
a_{z,c}+n_{z,c},
b_{z,c}+\Delta t
\right).
$$

The posterior mean used by the policy is therefore

$$
\hat{\lambda}_{z,c}
=
\frac{a_{z,c}+n_{z,c}}
     {b_{z,c}+\Delta t}.
$$

### Size-aware prior

The prior will be uniform **per pickup-capable graph node**, not uniform per cluster. A larger cluster therefore receives a larger expected request rate simply because it contains more possible pickup locations.

Let

$$
q_z=\frac{|V_z|}{\sum_j |V_j|}
$$

be the fraction of pickup-capable graph nodes belonging to cluster \(z\).

Let \(\bar{\Lambda}_c\) be the prior total arrival rate for priority class \(c\), estimated only from training/historical data and not from hidden information in the test instance.

Then the prior mean for cluster \(z\) is

$$
\mu_{z,c}=q_z\bar{\Lambda}_c.
$$

We use a size-aware Gamma prior without interpreting the prior as fabricated historical observations.

Let \(\eta>0\) control the concentration/strength of the prior and let \(\Lambda_c^0\) be the configured city-wide expected arrival rate for importance class \(c\), expressed in orders per minute. Then

$
a^{(0)}_{z,c}=\eta q_z,
\qquad
b^{(0)}_c=\frac{\eta}{\Lambda_c^0}.
$

Therefore

$
E[\lambda_{z,c}]
=
\frac{a^{(0)}_{z,c}}{b^{(0)}_c}
=
q_z\Lambda_c^0.
$

The key intuition is that **cluster size controls the prior mean**: if a cluster contains 20% of the pickup-capable graph nodes, then before observing online demand it receives 20% of the prior city-wide request rate for that importance level.

When an order arrives, only the matching \((z,c)\) event count increases. Elapsed simulation time is exposure for every \((z,c)\) pair. If \(n_{z,c}(t)\) matching orders have arrived by time \(t\), then

$
\lambda_{z,c}\mid data
\sim
\mathrm{Gamma}
\left(
a^{(0)}_{z,c}+n_{z,c}(t),
\;
b^{(0)}_c+t
\right).
$

The cluster-size term uses graph coverage size, not the number of historical HDBSCAN samples, so past demand is not counted twice.

## 4. Score for choosing which concrete robots to reserve

The FCNN determines **how many** robots of each type belong to each reservation stratum. We must then decide **which physical robots** are assigned to those strata.

For a reservation threshold \(c\), define the set of priorities protected by that threshold as \(\mathcal P(c)\).

First define the expected relevant workload in cluster \(z\):

$
W_{z,c}
=
\sum_{p\ge c}\hat{\lambda}_{z,p}.
$

Then normalize it into a demand weight:

$
w_{z,c}
=
\frac{W_{z,c}}{\sum_j W_{j,c}}.
$

For robot \(r\):

- \(a_r\): estimated time until the robot reaches its next state from which its route can be changed;
- \(u_r\): graph node of that state;
- \(m_z\): representative/medoid of cluster \(z\);
- \(v_r\): robot speed;
- \(\tilde d(u_r,m_z)\): Haversine distance between the graph-node coordinates;
- \(\tilde C(r,z)\): estimated charging delay needed to respond from that state to cluster \(z\).

The reservation score is

$$
H_{\mathrm{reserve}}(r,c)
=
\sum_z
w_{z,c}
\left[
a_r
+
\frac{\tilde d(u_r,m_z)}{v_r}
+
\tilde C(r,z)
\right].
$$

**Lower is better.**

Within each robot type, the robots with the smallest score are selected for the more restrictive high-priority reservation strata.

Concretely, the FCNN fractions are converted into integer robot counts for every type and priority threshold. Thresholds are processed from the largest importance value to the smallest. At threshold \(c\), the required number of currently unassigned type-\(k\) robots with the lowest \(H_{\mathrm{reserve}}(r,c)\) are assigned to that stratum. A robot assigned threshold \(c\) may serve only requests with importance \(p\ge c\). Remaining robots are assigned to the general-service stratum.

This score directly combines predicted spatial demand, current workload/availability, location, robot speed, and battery/charging state.

### Intuition for summing over all clusters

We do not score a robot only by the cluster it currently occupies. Cluster borders are artificial and robots are mobile. A robot may be physically just outside a high-demand cluster yet be much faster to reach its demand than a robot located inside that cluster but far from its active area.

The score is therefore a **demand-weighted expected response time**:

- high \(W_{z,c}\) means cluster \(z\) is expected to generate many requests of importance \(c\) or higher;
- such a cluster receives a large weight;
- a robot with small response time to that busy cluster receives a better (lower) total score;
- low-demand clusters have little influence on the score.

Importantly, we multiply response time by demand weight rather than divide by demand. Dividing by a small demand value would make nearly irrelevant clusters create very large penalties.

### Capacity safeguard

Reservation is not allowed to remove all large-capacity robots from general service.

At minimum, one robot capable of carrying the largest package class must remain available to all priority classes. This can later be generalized to a minimum open-capability constraint for every payload/volume class.

## 5. Cheap candidate-assignment score for a new request

For a newly arrived request \(j\), we do **not** want to run an exact shortest-path and battery-aware insertion search for every robot.

First, hard feasibility filters remove robots that cannot carry the package or are not allowed to serve its priority class.

For every remaining robot, we evaluate possible pickup/dropoff insertion positions using **cheap travel-time estimates** rather than exact route planning.

For any pair of nodes \(x,y\),

$$
\tilde{\tau}_r(x,y)
=
\frac{d_{\mathrm{hav}}(x,y)}{v_r},
$$

where \(d_{\mathrm{hav}}\) is the Haversine great-circle distance computed from the latitude and longitude stored on the two graph nodes:

$$
d_{\mathrm{hav}}(x,y)
=
2R\arcsin\!\left(
\sqrt{
\sin^2\!\left(\frac{\Delta\phi}{2}\right)
+
\cos(\phi_x)\cos(\phi_y)
\sin^2\!\left(\frac{\Delta\lambda}{2}\right)
}
\right).
$$

Haversine is used only in the cheap heuristic stage. It avoids a shortest-path query for every robot and insertion candidate. The final search over shortlisted robots still uses exact graph distances and exact battery-feasible routing.

For robot \(r\), let \(S_r\) be its current ordered stop sequence. For each precedence-feasible insertion of pickup \(p_j\) and delivery \(d_j\), construct an approximate sequence

$$
\tilde S_r^{(i,k)}.
$$

We propagate estimated completion times through that sequence using:

- approximate travel time;
- service time;
- current battery;
- energy consumption;
- estimated charging duration when the approximate energy trajectory requires charging.

No exact shortest-path or exact battery-routing search is performed at this stage.

For each current order \(o\) assigned to \(r\), and for the new order \(j\), use the same project loss function on the estimated delivery time.

The cheap assignment score is

$$
H_{\mathrm{assign}}(r,j)
=
\min_{i<k}
\left[
\sum_{o\in O_r\cup\{j\}}
L(\hat T_o^{(i,k)},D_o,w_o)
-
\sum_{o\in O_r}
L(\hat T_o^{\,0},D_o,w_o)
\right].
$$

Here:

- \(O_r\) is the set of orders currently assigned to robot \(r\);
- \(\hat T_o^{(i,k)}\) is the estimated delivery time after the candidate insertion;
- \(\hat T_o^0\) is the estimated delivery time before inserting the new order.

Therefore the heuristic explicitly estimates the **incremental loss caused by the assignment over all affected orders**, including delays to packages already being carried or scheduled.

This is essential because a robot can carry several orders at once. A robot may deliver the new order quickly but delay an existing high-importance order enough to create a much larger total penalty. Such a robot should receive a worse score even if the new order alone looks attractive.

Haversine distance is therefore only an input used to estimate completion times. The heuristic ranking criterion is \(H_{\mathrm{assign}}\), the estimated change in total loss, not distance and not the new request's delivery time.

**Lower is better.**

## 6. Top-\(K\) robot pruning

All eligible robots are ranked using

$$
H_{\mathrm{assign}}(r,j).
$$

Only the best \(K\) robots, or the best \(x\%\) of eligible robots, are passed to the expensive exact search.

This first-stage pruning is added on top of the exact lower-bound pruning and battery-routing optimizations already present in the project.

## 7. Tuning the shortlist

For small and medium instances where exhaustive evaluation is practical, full robot search provides the ground truth.

For different values of \(K\) or \(x\%\), we will measure

$$
\mathrm{Recall@K}
=
P(\text{globally best robot is contained in the heuristic top-}K)
$$

and

$$
\mathrm{Regret}
=
L_{\mathrm{shortlist}}
-
L_{\mathrm{full\ search}}.
$$

We will also measure wall-clock decision time.

The goal is to find a shortlist size that gives a large computational reduction with negligible degradation in solution quality.

## 8. Exact greedy selection on the shortlisted robots

The current final decision is intentionally **greedy with respect to all known orders at the current decision epoch**.

For each shortlisted robot \(r\), the planner performs the full exact insertion search and computes

$$
\Delta L_{\mathrm{exact}}(r,j)
=
L_{\mathrm{current\ orders\ after\ assigning}\ j\ \mathrm{to}\ r}
-
L_{\mathrm{current\ orders\ before}}.
$$

The chosen robot is

$$
r^*
=
\arg\min_{r\in C_K}
\Delta L_{\mathrm{exact}}(r,j),
$$

where \(C_K\) is the heuristic shortlist.

Thus, the shortlist is approximate, but the final choice **within the shortlist** is the true greedy minimum-loss decision under the current known set of orders.

For each shortlisted robot, the exact search considers all precedence-feasible pickup/dropoff insertion positions while preserving feasibility. Because a robot may already carry several orders, the exact score is the cumulative change in loss of **all orders affected by that robot's new route**, not the loss of the newly arrived order alone.

### Later predictive extension

A later version will move beyond purely current-order greedy loss.

The intended future objective is conceptually

$$
\Delta L_{\mathrm{current}}(r,j)
+
\gamma
E\left[
L_{\mathrm{future}}
\mid
\text{state after assigning }j\text{ to }r,
\hat{\lambda}
\right].
$$

Future requests can be generated or predicted using the learned spatial/priority demand model.

This predictive component is deliberately left for a later phase; the first version will optimize the exact loss of currently known orders only.

## 9. Experimental decomposition

The components will be evaluated separately so that their individual contributions can be identified.

Planned comparisons include:

- no priority reservation vs. learned reservation;
- homogeneous/global reservation vs. robot-type-specific reservation;
- original recent-demand reservation selection vs. the posterior spatial-demand reservation score;
- uniform-per-cluster prior vs. size-aware prior;
- HDBSCAN-based regions vs. an alternative full-partition clustering method;
- exhaustive robot evaluation vs. heuristic top-\(K\);
- different \(K\) / shortlist percentages;
- different prior strengths \(\tau_0\);
- different workload levels;
- different fleet heterogeneity levels;
- different graph structures;
- different demand and priority distributions;
- different levels of travel-time stochasticity.

## 10. Benchmark environments

The project will combine controlled synthetic graphs/workloads with real road graphs and historical or semi-real demand traces.

## 11. Simulation time vs. planning compute time

The main simulator is event-driven. When an event at simulated time \(t\) requires a planning/assignment decision, the simulated clock is held at \(t\) until the policy returns its decision.

Wall-clock computation time is measured separately.

This is the main benchmark convention because it keeps comparisons hardware-independent and reproducible.

Planning latency is nevertheless an important deployment metric, so a **later realism experiment** will explicitly inject measured computation time into the simulation:

1. planning begins at simulated time \(t\);
2. robots continue their already committed physical motion while the planner computes;
3. after the measured wall-clock planning delay, the action is applied to the state the system has actually reached.

The comparison between frozen-time and latency-aware simulation will show whether the instantaneous-planning assumption materially affects the conclusions.

## Status

The simulator currently supports heterogeneous robots, battery-aware routing, charging stations and queues, interruptible return-to-charge behavior, busy-robot scheduling, and reactive pickup/dropoff insertion.

Current implementation work on this branch includes:

1. offline HDBSCAN spatial regions with complete graph coverage and persisted `in_cluster` node labels;
2. the size-aware Gamma-Poisson model for every \((cluster, importance)\) pair;
3. online posterior updates and demand-weighted concrete-robot reservation;
4. reusable Haversine distance and cluster-response-time estimates for the cheap heuristic stage;
5. a joint FCNN with constrained per-type reservation fractions, offline target-selection tooling, model training, and model persistence;
6. deterministic largest-remainder conversion from fractions to robot counts, high-to-low physical-robot assignment, and the large-capacity general-service safeguard;
7. an unexecuted `run_nyc_heuristic_policy.py` simulation runner that updates the Gamma-Poisson posterior on each arrival, recomputes reservations, ranks eligible robots by Haversine-estimated all-order incremental loss, and applies exact incremental-loss selection inside the best \(K\). If every initial top-\(K\) robot is exactly battery-infeasible, the runner safely expands the ranking until it finds a feasible robot.

The runner leaves reservation disabled when no model is supplied, which provides the no-reservation ablation without a separate simulator. Enable it with:

```powershell
python scripts/run_nyc_heuristic_policy.py --reservation-model models/reservation_fcnn.npz
```

The predictive future-order objective, experiment-scale shortlist tuning/full ablation study, and latency-aware simulator remain later research experiments rather than missing components of the current greedy policy.

## Training and simulation on the faculty server

The expensive step is perfect-information target generation, because it runs one complete simulation for every `(training scenario, alpha candidate)` pair. It is process-parallel. The FCNN itself is intentionally small, so its parallelism is implemented as independent random restarts followed by validation-model selection.

### 1. Environment

```bash
cd /data/workspace/robot_delivery/dynamic-delivery-robot-fleet
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy networkx scikit-learn pytest
export PYTHONPATH="$PWD/src"
```

### 2. Prepare the training manifest

Copy `configs/reservation_training_manifest.example.json`, list **training-only** scenarios, and add the fixed heterogeneous alpha candidates to evaluate. Fraction rows follow ascending importance order `[1, 2, 5]` and must sum to one for every robot type.

The paper used 100 training instances and enumerated candidate alpha settings. For this heterogeneous extension, avoid the full Cartesian product across four robot types unless it is deliberately bounded: it grows exponentially. Use a designed candidate set or a reproducible sampled set that includes no-reservation, balanced, and stronger-reservation configurations.

### 3. Generate perfect-information targets in parallel

```bash
python scripts/generate_reservation_training_data.py \
  configs/reservation_training_manifest.json \
  data/training/reservation_targets.npz \
  --processes 128 \
  --work-dir data/training/reservation_target_jobs
```

`--processes` is the maximum number of simultaneous simulator evaluations. Completed job JSON files are cached in the work directory, so rerunning the same command resumes instead of repeating completed simulations.

Although the server has roughly 700 logical cores, each process loads a graph and routing index. With 300 GB RAM, begin with 128 processes, inspect memory consumption, and increase to 256 or higher only if there is comfortable headroom. The script forces numerical libraries to one thread per simulator process to avoid oversubscription.

### 4. Train parallel FCNN restarts

```bash
python scripts/train_reservation_fcnn.py \
  data/training/reservation_targets.npz \
  models/reservation_fcnn.npz \
  --processes 64 \
  --restarts 256 \
  --epochs 2000 \
  --learning-rate 0.01 \
  --validation-fraction 0.2 \
  --threads-per-process 1
```

`--processes` controls simultaneous training processes; `--restarts` controls the total independent initializations. The best validation seed is retrained on the complete training set and saved. A companion `models/reservation_fcnn.training.json` records the selected seed and losses. The paper-default hidden width is used when `--hidden-dim` is omitted or set to `0`.

### 5. Run the proposed simulation

```bash
python scripts/run_nyc_heuristic_policy.py \
  --graph data/graphs/new_york_city.graphml \
  --scenario data/scenarios/nyc_reference_12h_seed42.json \
  --reservation-model models/reservation_fcnn.npz \
  --shortlist-k 10 \
  --importance-prior-rates-per-hour '{"1": 120.0, "2": 22.5, "5": 7.5}' \
  --prior-concentration 4.0 \
  --output results/proposed_policy.json
```

For the no-reservation ablation, omit `--reservation-model`. For an offline fixed-alpha evaluation, use `--fixed-reservation-fractions path/to/fixed_alpha.json`; this option is intended for target generation, not the final online policy.

### 6. Verify before large runs

```bash
pytest -q
python scripts/run_nyc_heuristic_policy.py --help
python scripts/generate_reservation_training_data.py --help
python scripts/train_reservation_fcnn.py --help
```
