# Dynamic Delivery Robot Fleet

Course project for **AI and Autonomous Systems**.

We study online planning and assignment for a heterogeneous fleet of autonomous delivery robots operating on a city graph under stochastic demand and travel times.

## Core setting

- Pickup-to-delivery requests arrive online.
- Each request has a pickup, destination, package size/weight, importance level, request time, and deadline.
- Robots are heterogeneous and may differ in speed, payload/volume capacity, battery capacity, and energy consumption.
- Robots may already have assigned packages, so a new request may be inserted into an existing route.
- The planner may wait for a better robot instead of assigning immediately.
- Future extensions include package transfer between robots and proactive repositioning.

## Primary objective

The current benchmarks use a deadline- and importance-aware delivery loss. For a request with request-to-delivery time \(T\), allowed duration \(D\), and importance \(w\):

$$
L(T,D,w)=w\min(D,T)+(w+1)^2\max(0,T-D).
$$

The first term rewards shorter delivery time before the deadline, while lateness receives a much larger importance-dependent penalty.

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

The intended model will predict all robot-type reservation vectors jointly, so that the outputs remain mutually consistent.

Likely additional input features for our setting include robot-type workload, availability, battery statistics, spatial demand estimates, and time within the horizon.

**Implementation and training of the FCNN are intentionally deferred to a later phase.**

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
- \(\tilde d(u_r,m_z)\): cheap/precomputed distance estimate;
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
\frac{\tilde d(x,y)}{v_r},
$$

where \(\tilde d\) is an \(O(1)\) or precomputed approximation to graph distance. One candidate implementation is geographic distance multiplied by a calibrated graph-stretch factor, potentially estimated separately for pairs of spatial regions.

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
3. demand-weighted reservation scoring infrastructure.

The next coding steps are the cheap all-order incremental-loss candidate heuristic, top-\(K\) pruning integration, and the final reservation-policy wiring.

The FCNN training procedure, predictive future-order objective, full ablation study, and latency-aware simulator are intentionally deferred to later phases.
