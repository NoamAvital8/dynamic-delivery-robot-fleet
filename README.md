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
7. a tested `run_nyc_heuristic_policy.py` simulation runner that updates the Gamma-Poisson posterior on each arrival, recomputes reservations, ranks eligible robots by Haversine-estimated all-order incremental loss, and applies exact incremental-loss selection inside the best \(K\). If every initial top-\(K\) robot is exactly battery-infeasible, the runner safely expands the ranking until it finds a feasible robot;
8. an optional predictive idle runner that coordinates staying, demand-aware repositioning, and full background charging.

The runner leaves reservation disabled when no model is supplied, which provides the no-reservation ablation without a separate simulator. Enable it with:

```powershell
python scripts/run_nyc_heuristic_policy.py --reservation-model models/reservation_fcnn.npz
```

### Predictive idle control on top of the greedy dispatcher

`scripts/run_nyc_anticipatory_idle_policy.py` uses the same greedy order assignment and replaces the fixed nearest-charger idle rule. At each idle decision epoch it scores `STAY`, moves to a bounded set of HDBSCAN representatives, and full charging at a bounded set of reachable stations. It uses only the Gamma-Poisson posterior at the current simulation time; no held-out future orders enter the decision.

For cluster \(z\), importance \(c\), and a short horizon \(H\), the predicted demand value is \(W_{z,c}=H\hat\lambda_{z,c}(c+1)^2\). For robot \(r\) under action \(a\), its approximate response coverage is

\[
g_{r,z,c}(a)=\kappa_r\,b_{r,z}(a)\,
e^{-d_{hav}(y_r(a),z)/(60v_r\tau_c)}
\frac1H\int_0^H e^{-(t^{ready}_r(a)-u)_+/\tau_c}\,du,
\qquad \tau_c=18/\sqrt c,
\]

where \(u\) is a future order's arrival time, \(\kappa_r\) reflects carrying capability, and \(b_{r,z}\in[0,1]\) reflects battery remaining after reaching the region. A move's initial unavailability is averaged over the horizon, so orders arriving after the move benefit from the robot's new location. `18` is the configurable response-time scale in minutes. The fleet utility is

\[
U(a_1,\ldots,a_R)=\sum_{z,c}W_{z,c}\left(1-e^{-\sum_r g_{r,z,c}(a_r)}\right).
\]

The concave term gives diminishing returns: an additional robot moves to an already-covered region only when its *marginal* gain exceeds the gain elsewhere and the minimum-movement threshold. Idle actions are selected sequentially and the projected fleet coverage is updated after each choice. `STAY` is the baseline, so travel is selected only for positive marginal utility.

Charging candidates include exact travel energy, current port occupancy, FIFO requests, in-transit charging plans, and the earliest non-overlapping interval on a physical port. The projected ready time includes travel, waiting, and charging. Every selected background charge targets **100% battery**; the idle planner never interrupts an active charge for another idle action. A real delivery assignment may still interrupt it. Charging plans are rebuilt after changed events, and assignment cancels a robot's pending background travel intent.

The score is a bounded one-step surrogate for expected reduction in future delivery loss. It is **not** the proposed full Monte Carlo Bellman rollout. That remains a later experiment after this controller is measured against the fixed idle baseline. The runner reports idle action counts and planning wall time alongside total delivery loss.

Online work is bounded by the top 32 demand clusters, at most 5 relocation and 3 charging candidates per idle robot, and at most 32 selected movements per planning epoch. Replanning defaults to every 15 simulated minutes. Candidate response vectors use a reusable process pool when a batch contains at least 64 vectors; `--idle-processes` controls its size. The graph and routing index stay in the simulation process and are not copied to workers.

Busy-robot coverage uses the estimated completion of its current itinerary, not
a fixed number of minutes per service stop. It sums remaining Haversine legs
divided by robot speed, unfinished pickup/dropoff handling, planned charging
energy divided by charger power, and expected waits from the current charging
port calendar. Cached route plans supply charger waypoints; no new exact route
search is needed for this estimate. The committed edge and any current charge,
queue, or handling operation are counted only for their remaining duration.
Coverage is evaluated at the final service location and estimated remaining
battery. Future unobserved queue arrivals and route interruptions are not known,
so this remains an approximation. An incomplete or energy-inconsistent itinerary
is marked unavailable and counted in result diagnostics instead of inventing
battery or a fixed delay. Results identify this estimator as
`haversine_route_handling_charge_queue_v1`. Existing reservation NNs are reused;
rerun affected idle-policy evaluations in a new results directory.

### Conservative relocation under uncertain demand

Add `--idle-relocation-uncertainty-penalty 1.0` to the anticipatory idle runner to require stronger evidence before a robot leaves its current location. The original posterior-mean planner remains available with the default value `0.0`.

For each relocation, let \(d_{z,c}=e^{-G_{z,c}}(1-e^{-\Delta g_{z,c}})\), where \(G\) is current fleet coverage and \(\Delta g\) is the proposed action's coverage change relative to staying. Under the independent Gamma posteriors \(\lambda_{z,c}\sim\mathrm{Gamma}(\alpha_{z,c},\beta_{z,c})\), the existing expected marginal gain and its posterior standard deviation are

\[
\mu=\sum_{z,c}H(c+1)^2d_{z,c}\frac{\alpha_{z,c}}{\beta_{z,c}},
\qquad
\sigma=\sqrt{\sum_{z,c}\left[H(c+1)^2d_{z,c}\right]^2
\frac{\alpha_{z,c}}{\beta_{z,c}^2}}.
\]

The relocation score is \(\mu-\rho\sigma\), where \(\rho\) is the configured penalty. A move must exceed the same minimum-gain threshold as before. Negative coverage changes contribute to uncertainty too, so the policy accounts for uncertain demand near the robot's current location as well as uncertain demand at its destination. Scores are recomputed after each selected fleet action. Charging retains its original mean-based score, feasibility checks, port scheduling, and full-battery target.

This penalty measures uncertainty in the demand **rate**; it does not include the additional randomness of future Poisson arrivals. It is a conservative surrogate score, not a calibrated confidence bound or a direct prediction of actual delivery loss. No city-specific threshold is assumed, and improvement must be established on held-out scenarios.

The multicity runner adds two opt-in policies: `full_uncertainty_idle` and `full_uncertainty_idle_no_nn`. Its default six policies remain the original comparison. Once the spatial reservation model is trained, run the paired comparison in a new results directory:

```bash
python scripts/run_multicity_experiments.py \
  runs/icaps_multicity_v1/suite/suite.json \
  runs/idle_uncertainty_v1 \
  --spatial-model runs/icaps_multicity_v1/training/spatial/reservation_fcnn.npz \
  --paper-model runs/icaps_multicity_v1/training/paper/reservation_fcnn.npz \
  --processes 4 --idle-processes 1 \
  --idle-relocation-uncertainty-penalty 1.0 \
  --policies full full_uncertainty_idle full_no_idle full_no_nn full_uncertainty_idle_no_nn
```

Compare paired delivery loss, on-time/late deliveries, and runtime. `summary.csv` also reports relocation counts, charging counts, stay decisions, idle-planning time, and the uncertainty penalty. These settings are recorded in results and benchmark fingerprints; changing the penalty invalidates the corresponding cached variant results.

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
  --processes 24 \
  --work-dir data/training/reservation_target_jobs
```

`--processes` is the maximum number of simultaneous simulator evaluations. Completed job JSON files are cached in the work directory, so rerunning the same command resumes instead of repeating completed simulations.

The faculty VM reports **72 logical CPUs (36 physical cores) and 373 GiB RAM**. Each target-generation process loads a graph and routing index, so start with 24 concurrent simulations and inspect memory and throughput before increasing the count. The script forces numerical libraries to one thread per simulator process to avoid oversubscription.

### 4. Train parallel FCNN restarts

```bash
python scripts/train_reservation_fcnn.py \
  data/training/reservation_targets.npz \
  models/reservation_fcnn.npz \
  --processes 16 \
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

Run the predictive idle controller with the same graph, scenario, and optional reservation model:

```bash
python scripts/run_nyc_anticipatory_idle_policy.py \
  --graph data/graphs/new_york_city.graphml \
  --scenario data/scenarios/nyc_reference_12h_seed42.json \
  --shortlist-k 10 \
  --idle-processes 8 \
  --idle-horizon-min 45 \
  --idle-replan-interval-min 15 \
  --output results/anticipatory_idle.json
```

Add `--reservation-model models/reservation_fcnn.npz` when that trained model is available. Compare its `loss_objective` with the fixed-idle `run_nyc_heuristic_policy.py` result using the same scenario and assignment parameters.

### 6. Verify before large runs

```bash
pytest -q
python scripts/run_nyc_heuristic_policy.py --help
python scripts/run_nyc_anticipatory_idle_policy.py --help
python scripts/generate_reservation_training_data.py --help
python scripts/train_reservation_fcnn.py --help
```

## Resumable five-city ICAPS comparison

`scripts/run_multicity_campaign.py` prepares HDBSCAN **leaf** partitions, generates
20 training and 5 disjoint test scenarios per city (100/25 total), trains one
joint reservation NN across all five cities, trains a separate moving-average
NN for the paper-adapted comparator, and evaluates six policies on the same
held-out scenarios. The paper-adapted comparator shares our battery-aware,
heterogeneous simulator; it uses the paper's recent-demand moving-average
features and recent-pickup vehicle-reservation ranking, so it is an *adaptation*,
not a reproduction of the original homogeneous-fleet experiments.

The six rows are: full posterior-NN + predictive idle; battery-feasible earliest
completion (Myopic A+B); exhaustive reactive incremental-loss insertion;
paper-adapted moving-average reservation; full minus NN reservation; and full
minus predictive idle control. Main outcomes are total delivery loss, on-time
and late deliveries, and both wall-clock runtime and simulated finish time.

On the VM, launch a detached campaign so it survives closing the SSH session
or turning off the local PC (assuming the VM itself remains running):

```bash
cd /data/workspace/robot_delivery/dynamic-delivery-robot-fleet
mkdir -p runs/icaps_multicity_v1
nohup .venv/bin/python -u scripts/run_multicity_campaign.py \
  runs/icaps_multicity_v1 \
  --max-processes 60 --target-processes 60 --nn-processes 60 \
  --restarts 256 --benchmark-processes 12 --idle-processes 4 \
  > runs/icaps_multicity_v1/campaign.log 2>&1 < /dev/null &
echo $!
```

The VM reports 72 logical CPUs. `--max-processes 60` is a ceiling, not a
promise to launch 60 large simulator jobs: label generation and benchmarks
also respect a default 8 GiB/job estimate and 75% of currently available RAM.
Benchmark concurrency reserves room for each job's idle-planning workers.
The effective counts are printed as `RESOURCE_BUDGET` in `campaign.log`.
Changing only concurrency settings when resuming an existing campaign is
allowed; the run records the resource history and keeps scientific settings
fixed. Numerical-library threads are capped at one per worker.

`runs/icaps_multicity_v1/campaign_status.json` names the current/failed stage;
`runs/icaps_multicity_v1/stages/*.log` contain stage logs. Every simulator run
gets its own `result.json`, `result.progress.json`, `runner.log`, and
`status.json` under `runs/icaps_multicity_v1/benchmarks/<city>/<scenario>/<policy>/`.
The rolling `benchmarks/summary.csv` has one row per city, test seed, and
policy; `benchmarks/city_policy_summary.csv` sums loss, runtime, on-time and
late deliveries across completed seeds for each city and algorithm. A failed
simulation retains its last progress snapshot and log;
completed training evaluations and benchmark rows are cached with input
fingerprints. Re-run the campaign with the same run directory and scientific
settings to resume after a failure; resource counts may change.
Training-label and benchmark logs print completed/total, elapsed time, and a
throughput-based ETA. Individual simulator progress snapshots update about
every 30 seconds. Each NN initialization is checkpointed separately, so a
crash during its 256 restarts does not discard completed restarts. The final
all-data model fit is short and may be repeated after a crash. A one-time
`--compatible-source-fingerprint <sha256>` migration is available for
completed simulator outputs from a verified earlier code version; use it only
when the code change did not alter successful jobs' results.

Cluster quality is a preflight gate: the campaign refuses to train if the
largest HDBSCAN region covers more than half a city's graph nodes. Training
and test scenarios never overlap, and city-specific prior rates come only
from each city's training scenarios. The two NNs are each shared across all
cities, not trained separately per city.

### Tiny proven-optimum comparison

The full city-scale dynamic problem has no practical globally exact solver.
For a diagnostic lower bound, `scripts/run_small_optimal_comparison.py` creates
two-robot, four-order cases and exhaustively enumerates every robot assignment
and precedence-valid pickup/drop-off route. A conservative certificate proves
that battery and payload cannot bind on these tiny cases, so no charging
detour can improve the loss. The offline oracle knows future arrivals; it is
not a deployable online policy. The default run compares it with Myopic A+B
and reactive insertion, saving routes and loss gaps:

```bash
.venv/bin/python scripts/run_small_optimal_comparison.py \
  runs/tiny_oracle_v1 --scenarios 3 --processes 6
```

Once the two trained NN files exist, use a **new run directory** (for example
`runs/tiny_oracle_six_v1`) and add `--policies full myopic_ab
reactive_insertion paper_sa_adapted full_no_nn full_no_idle` with
`--spatial-model <path>` and `--paper-model <path>` to compare all six.
The new directory matters because benchmark cache signatures include the
model inputs; do not reuse the two-policy directory with different inputs.
The oracle results are in `runs/tiny_oracle_v1/oracle/`; the paired policy
results and `oracle_comparison.csv` are in the same run directory. Every
oracle result and policy result is independently resumable. The five test
seeds per city in the current campaign are a pilot; a publication-strength
claim should use more paired test scenarios and explicit demand regimes.
