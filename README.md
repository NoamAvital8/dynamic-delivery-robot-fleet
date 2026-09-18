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

The current benchmarks use a deadline- and importance-aware delivery loss. For a request with request-to-delivery time `T`, allowed duration `D`, and importance `w`:

\[
L(T,D,w)=w\min(D,T)+(w+1)^2\max(0,T-D).
\]

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

## Proposed anticipatory algorithm

The proposed method extends the capacity-reservation idea from the scalable anticipatory DPDP policy to a **heterogeneous robot fleet**.

At every decision epoch, the policy performs two main tasks:

1. determine how much capacity of each robot type should be reserved for high-importance requests;
2. assign the newly arrived request using a fast heuristic shortlist followed by exact loss evaluation.

The intended pipeline is:

[
	ext{online demand estimation}
ightarrow
	ext{FCNN reservation policy}
ightarrow
	ext{select concrete reserved robots}
ightarrow
	ext{cheap assignment heuristic}
ightarrow
	ext{top-}K	ext{ candidate robots}
ightarrow
	ext{exact incremental-loss evaluation}.
]

### 1. Heterogeneous reservation fractions

Instead of learning one global reservation vector for a homogeneous fleet, the FCNN predicts a reservation distribution for each robot type.

For robot type (k) and importance threshold (c),

[
alpha_{k,c}
]

denotes the fraction of robots of type (k) that should be restricted to requests of importance (c) or higher.

A single model should predict the reservation vectors jointly so that the decisions for the different robot types and importance levels remain consistent.

Candidate FCNN inputs include:

- estimated demand by importance level;
- total workload per robot;
- current backlog by importance;
- fraction of robots busy/idle by type;
- battery-state statistics;
- current spatial demand estimates;
- time within the planning horizon.

Training is performed offline. For generated training instances, candidate reservation configurations are evaluated using simulation, and the best-performing configuration is used as the supervised target.

### 2. Online spatial demand estimation

Reservation fractions determine **how many** robots should be protected, but not **which physical robots** should be selected.

Pickup locations are therefore divided into spatial demand regions, initially using a method such as DBSCAN. For each cluster (z) and importance class (c), the policy maintains an arrival-rate estimate

[
lambda_{z,c}.
]

A Gamma-Poisson Bayesian model is used:

[
lambda_{z,c}sim mathrm{Gamma}(a_{z,c},b_{z,c}).
]

After observing (n) requests in the cluster during an exposure interval (Delta t),

[
lambda_{z,c}mid data
sim
mathrm{Gamma}(a_{z,c}+n,;b_{z,c}+Delta t).
]

This provides a stable online estimate even for rare high-importance requests and naturally implements a prior-to-posterior update.

Demand may change over time, so a forgetting/window mechanism can later be added if needed.

### 3. Topology-informed spatial prior

The prior should not simply assume that graph centrality equals demand.

Instead, graph topology may be used to create a **weak prior** for regional demand. Candidate prior features include:

- closeness/accessibility;
- node density;
- average shortest-path distance to the service region;
- local connectivity;
- betweenness centrality.

Observed request data should dominate the prior as evidence accumulates.

The experiments will compare at least:

- a uniform/weak prior;
- a topology-informed prior.

This will determine whether graph structure actually helps estimate demand rather than assuming that it does.

### 4. Selecting the concrete robots to reserve

Once the FCNN predicts the required reservation fractions, the policy must choose the actual robots assigned to each importance threshold.

A reservation heuristic will rank robots according to how well positioned they are for predicted high-importance demand.

Conceptually,

[
H_{mathrm{reserve}}(r,c)
approx
	ext{expected response time of robot }r
	ext{ to predicted demand with importance }ge c.
]

The score should use the posterior regional demand estimates as weights and may consider:

- the robot's expected usable location;
- robot type and speed;
- current workload;
- battery state and charging needs;
- predicted high-priority demand near that location.

The exact form of this heuristic is still to be finalized and will be evaluated empirically.

A feasibility safeguard is also required: reservation must never make every large-capacity robot unavailable to lower-priority requests. At least one robot capable of serving the largest package class remains generally accessible. This can later be generalized to a minimum open-capacity constraint per package-size class.

### 5. Fast heuristic candidate selection for a new request

Evaluating exact insertion loss for every robot is expensive because each robot may have several possible pickup/dropoff insertion positions and battery-feasible routes.

The proposed policy therefore first computes a **cheap estimate** for every eligible robot without running a full shortest-path/insertion search for every ((robot, pickup, destination)) combination.

The heuristic estimates the time-related consequences of assigning the new request:

[
hat T
=
	ext{existing workload}
+
	ext{estimated travel time}
+
	ext{estimated charging time}
+
	ext{service time}.
]

The estimate should account for:

- approximate robot-to-pickup and pickup-to-destination travel time;
- robot speed;
- battery level and capacity;
- energy consumption;
- estimated charging duration;
- additional waiting time introduced to packages already assigned to the robot.

The heuristic then estimates the same quantity that the exact algorithm ultimately minimizes:

[
widehat{Delta L}_r
=
widehat{L}_{after}
-
widehat{L}_{before}.
]

Thus, the heuristic is not merely a nearest-robot rule; it is an inexpensive approximation to the true incremental objective.

### 6. Candidate pruning and exact evaluation

Robots are ranked by the cheap heuristic.

Only the best (K) robots, or the best (x%) of eligible robots, are passed to the expensive exact search.

For these shortlisted robots, the planner performs the full insertion evaluation, including:

- feasible pickup/dropoff insertion positions;
- graph routing;
- battery feasibility;
- charging time;
- effects on already-assigned packages;
- deadlines and importance-weighted loss.

The selected robot is

[
r^*
=
argmin_{rin	ext{CandidateSet}}
Delta L_r.
]

The existing exact lower-bound pruning and cached/vectorized battery-routing optimizations remain useful inside this shortlisted search.

### 7. Tuning the shortlist

The shortlist size is a parameter to be tuned experimentally.

For small or medium instances where exhaustive robot evaluation is affordable, the full search provides ground truth.

For different values of (K) or (x%), we will measure:

[
mathrm{Recall@K}
=
P(	ext{globally best robot is contained in the heuristic top-}K),
]

and

[
mathrm{Regret}
=
L_{mathrm{shortlist}}
-
L_{mathrm{full search}}.
]

The main trade-off is therefore:

[
	ext{planning runtime}
leftrightarrow
	ext{decision quality}.
]

The objective is to retain nearly all of the solution quality while evaluating only a small fraction of the fleet exactly.

## Experimental decomposition

The proposed components will also be evaluated separately so that the contribution of each part can be identified.

Planned comparisons include:

- no priority reservation vs. learned reservation;
- homogeneous/global reservation vs. robot-type-specific reservation;
- original recent-demand vehicle selection vs. spatial demand-aware reservation;
- uniform prior vs. topology-informed prior;
- exhaustive robot evaluation vs. heuristic top-(K);
- different values of (K) / shortlist percentage;
- different workload levels;
- different fleet heterogeneity levels;
- different graph structures;
- different demand and importance distributions;
- different levels of travel-time stochasticity.

## Benchmark environments

The project will combine controlled synthetic graphs/workloads with real road graphs and historical or semi-real demand traces.

## Simulation time vs. planning compute time

The simulator is event-driven. When an event at simulated time `t` requires a planning/assignment decision, the simulated clock is held at `t` until the policy returns its decision. Wall-clock computation time is measured separately and does **not** make robots move forward in simulated time.

This convention is intentional:

- **Hardware-independent comparisons.** Our main experimental question is whether one planning policy makes better routing/assignment decisions than another. If CPU time advanced the simulated world, the same algorithm could obtain a better delivery score simply by running on a faster machine, which would mix algorithm quality with hardware speed.
- **Reproducibility.** Freezing simulated time gives the same simulated trajectory when the same deterministic policy/scenario is run on a laptop, a CI runner, or a more powerful server.
- **Deployment compute can be provisioned independently.** A production fleet may use optimized and parallel hardware, while robot trips occur on a much slower physical timescale than planning computations.
- **Planning latency is still reported.** Wall-clock runtime and decision/search statistics are recorded rather than treating computation as free.

A separate realism experiment will inject measured planning latency into the simulation. During planning, robots continue their already committed motion, and the resulting decision is applied to the state reached when computation finishes.

This allows the project to test directly whether the instantaneous-decision assumption materially changes system performance.

## Status

The simulator currently supports heterogeneous robots, battery-aware routing, charging stations and queues, interruptible return-to-charge behavior, busy-robot scheduling, and reactive pickup/dropoff insertion. Benchmark and runtime optimization work is ongoing.

The next research phase is the implementation and evaluation of the heterogeneous reservation policy, online spatial-demand model, reservation heuristic, and top-(K) robot candidate pruning described above.
