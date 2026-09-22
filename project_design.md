# Project Design — Dynamic Delivery Robot Fleet

## 1. Project goal

We study online pickup-and-delivery planning for a heterogeneous fleet of autonomous delivery robots on a city graph.

Orders arrive dynamically. The planner must decide which robot should receive each order and where the new pickup/drop-off should be inserted into that robot's existing route.

A robot may carry and serve **multiple orders at the same time**. Therefore, assignment decisions are not evaluated only by the newly arrived order: they must account for how the new insertion changes the loss of **all active orders affected on that robot**.

The proposed policy combines:

**online spatial demand estimation → priority reservation → cheap Haversine candidate scoring → top-(K) shortlist → exact cumulative-loss insertion search**

---

## 2. Environment

### Orders

Each order (j) contains:

- pickup node (p_j);
- drop-off node (d_j);
- request time;
- package weight;
- package volume;
- importance (w_j);
- deadline / allowed delivery duration.

A larger numerical importance value means a more important order.

### Robots

The fleet is heterogeneous. Robots may differ in:

- speed;
- payload capacity;
- volume capacity;
- battery capacity;
- energy consumption.

A robot can have several active orders simultaneously. A new order can therefore be inserted before, between, or after the robot's existing pickup/drop-off stops, subject to feasibility.

### Battery and charging

Routing must respect:

- current battery state;
- energy consumption;
- charging-station locations;
- charging duration;
- charging queues in the exact simulator.

---

## 3. Loss function

For an order with:

- (T) = request-to-delivery time;
- (D) = allowed request-to-delivery duration;
- (w) = importance;

the loss is

[
L(T,D,w)
=
wmin(D,T)
+
(w+1)^2max(0,T-D).
]

### Intuition

- Before the deadline, every minute costs (w).
- After the deadline, the penalty becomes much larger: ((w+1)^2) per late minute.
- The policy therefore minimizes **loss**, not delivery time alone.
- Delaying an already-assigned important order can be worse than delivering the newly arrived order quickly.

---

# Part A — Anticipatory demand model

## 4. Offline spatial clustering

We use **HDBSCAN** to divide the graph into spatial demand regions.

HDBSCAN is useful because it can discover regions with different spatial densities without selecting one global DBSCAN radius.

### Problem

HDBSCAN may classify some nodes as noise, so HDBSCAN alone does not guarantee that every graph node belongs to a cluster.

### Final clustering procedure

This is performed **offline before the simulation runs**:

1. Run HDBSCAN on the graph-node coordinates.
2. Choose one representative node for each HDBSCAN cluster.
3. Keep all normal HDBSCAN cluster assignments.
4. For every HDBSCAN noise node, assign it to the nearest cluster representative using graph shortest-path distance.
5. Store the result directly on each graph node:

```text
in_cluster = cluster_id
```

Thus:

[
orall vin V,qquad cluster(v)	ext{ is defined}.
]

### Runtime consequence

The expensive clustering and graph-distance coverage work is done only during graph initialization.

During the online simulation, determining a node's cluster is an (O(1)) attribute lookup.

---

## 5. Demand distribution for every cluster and importance

For every pair

[
(z,c)
]

where

- (z) = spatial cluster;
- (c) = importance level;

we maintain an unknown order-arrival rate

[
lambda_{z,c}.
]

We model it with a Gamma-Poisson model:

[
lambda_{z,c}
sim
operatorname{Gamma}
left(
alpha^{(0)}_{z,c},
eta^{(0)}_c
ight),
]

using the **shape-rate** Gamma parameterization.

---

## 6. Size-dependent Gamma prior

The prior demand should be positively correlated with cluster size.

Define:

- (|V_z|) = number of graph nodes assigned to cluster (z);
- (|V|) = total number of graph nodes;
- (q_z = |V_z|/|V|) = fraction of the graph contained in cluster (z);
- (Lambda_c^0) = configured prior city-wide arrival rate for importance (c), in orders per minute;
- (eta>0) = prior concentration / confidence parameter.

Then:

[
oxed{
alpha^{(0)}_{z,c}
=
eta q_z
}
]

and

[
oxed{
eta^{(0)}_c
=
rac{eta}{Lambda_c^0}
}
]

so the prior mean is

[
oxed{
E[lambda_{z,c}]
=
rac{alpha^{(0)}_{z,c}}
     {eta^{(0)}_c}
=
q_zLambda_c^0.
}
]

### Intuition

If a cluster contains 20% of the pickup-capable graph nodes, then before seeing online observations it receives 20% of the prior city-wide rate for that importance class.

The prior is therefore:

- not uniform per cluster;
- positively related to cluster size;
- independent of the hidden future realization of the test scenario.

---

## 7. Online posterior update

Let:

- (n_{z,c}(t)) = number of orders observed by time (t) in cluster (z) with importance (c);
- (t) = elapsed observation time.

Then:

[
oxed{
lambda_{z,c}mid data
sim
operatorname{Gamma}
left(
alpha^{(0)}_{z,c}+n_{z,c}(t),
;
eta^{(0)}_c+t
ight)
}
]

and the posterior mean is

[
oxed{
hatlambda_{z,c}(t)
=
rac{
alpha^{(0)}_{z,c}+n_{z,c}(t)
}{
eta^{(0)}_c+t
}.
}
]

### Update rule

When a new order arrives:

- find the cluster containing its pickup node;
- read its importance;
- increment only the matching ((cluster, importance)) event count;
- elapsed time contributes exposure to every pair.

The posterior is then used as the current demand estimate.

---

# Part B — Priority reservation

## 8. FCNN reservation fractions

The reservation concept is based on the anticipatory policy of Ghiani et al.

The FCNN will eventually predict reservation fractions for combinations of:

- robot type (k);
- importance threshold (c).

Conceptually,

[
alpha_{k,c}
]

determines how much of robot type (k) should be protected for requests of importance (c) or higher.

Because reservation decisions interact, the intended design is one joint model rather than independent networks.

Possible FCNN inputs include:

- predicted demand by importance;
- expected requests per robot;
- active backlog by importance;
- busy fraction by robot type;
- battery statistics;
- spatial demand features;
- charger congestion;
- current time / remaining horizon.

### Status

**FCNN creation, training, target generation, and final output constraints are deferred to a later phase.**

---

## 9. Choosing which physical robots to reserve

After the FCNN determines how many robots should belong to a reservation stratum, we still need to choose the actual robots.

For an importance threshold (c), define the relevant predicted workload in cluster (z):

[
oxed{
W_{z,c}
=
sum_{pge c}
hatlambda_{z,p}.
}
]

Variables:

- (z) = cluster;
- (c) = reservation importance threshold;
- (p) = an importance level;
- (hatlambda_{z,p}) = posterior expected arrival rate in cluster (z) for importance (p);
- (W_{z,c}) = total expected rate in cluster (z) for importance (c) or higher.

Normalize:

[
oxed{
w_{z,c}
=
rac{W_{z,c}}
{sum_j W_{j,c}}.
}
]

Thus (w_{z,c}) tells us how much of the future (c+) demand is expected to come from cluster (z).

---

## 10. Robot response time to a cluster

For robot (r) and cluster (z), define:

[
oxed{
T_{r,z}
=
t_r^{avail}
+
t_{r,z}^{travel}
+
t_{r,z}^{charge}.
}
]

Variables:

- (t_r^{avail}) = estimated time until robot (r) reaches the next state from which it can respond;
- (x_r^{avail}) = robot's estimated usable location at that time;
- (m_z) = representative node of cluster (z);
- (v_r) = robot speed;
- (t_{r,z}^{travel}) = cheap estimated travel time from (x_r^{avail}) to (m_z);
- (t_{r,z}^{charge}) = estimated charging delay needed for the response.

For the cheap travel estimate:

[
oxed{
t_{r,z}^{travel}
=
rac{
d_{mathrm{hav}}(x_r^{avail},m_z)
}{
v_r
}.
}
]

Here (d_{mathrm{hav}}) is **Haversine distance**.

---

## 11. Reservation score

The reservation score is

[
oxed{
H_{mathrm{reserve}}(r,c)
=
sum_z
w_{z,c}T_{r,z}
=
rac{
sum_z W_{z,c}T_{r,z}
}{
sum_z W_{z,c}
}.
}
]

**Lower is better.**

### Intuition

This is the demand-weighted expected response time of robot (r) to future orders of importance (c) or higher.

We sum over **all clusters**, not only the cluster containing the robot.

Why:

- cluster boundaries are artificial;
- a robot can be near another cluster even if it is not technically inside it;
- a robot's current location may not be its future usable location;
- a robot near a busy cluster is more useful than a robot deep inside a quiet cluster.

A busy cluster gets a large (W_{z,c}), so response time to that cluster strongly influences the score.

We **multiply** response time by demand weight. We do not divide by demand, because dividing by a tiny demand value would make an almost irrelevant cluster create a huge penalty.

### Example intuition

Suppose:

- Cluster A expects 10 important orders/hour.
- Cluster B expects 2 important orders/hour.

A robot that reaches A in 3 minutes and B in 10 minutes should generally be preferred over a robot that reaches A in 7 minutes and B in 1 minute, because A is where most relevant demand is expected.

---

## 12. Capacity safeguard

Reservation must not make the fleet unable to serve large packages.

Initial rule:

[
oxed{
	ext{At least one largest-capacity robot remains available to general service.}
}
]

This can later be generalized into minimum open-capability constraints for multiple payload/volume classes.

---

# Part C — Cheap candidate-robot selection

## 13. Goal

When a new order arrives, we do **not** want to run a complete shortest-path, battery-routing, and exact insertion search for every robot.

Instead:

1. remove obviously infeasible robots;
2. cheaply estimate the incremental loss for each remaining robot;
3. rank them;
4. keep only the best (K);
5. run the exact algorithm only on those robots.

---

## 14. Hard feasibility filtering

Before heuristic scoring, remove robots that cannot legally serve the new request.

Examples:

- insufficient payload capacity;
- insufficient volume capacity;
- incompatible reservation threshold;
- no legal decision state.

This avoids wasting heuristic and exact computation.

---

## 15. Cheap distance heuristic: Haversine

The cheap stage uses **Haversine distance**, not graph shortest-path distance.

For nodes (x) and (y):

[
oxed{
d_{mathrm{hav}}(x,y)
=
2R
arcsin
left(
sqrt{
sin^2left(rac{Deltaphi}{2}ight)
+
cos(phi_x)cos(phi_y)
sin^2left(rac{Deltalambda}{2}ight)
}
ight).
}
]

Variables:

- (R) = Earth radius;
- (phi_x,phi_y) = node latitudes;
- (Deltaphi) = latitude difference;
- (Deltalambda) = longitude difference.

Estimated travel time for robot (r):

[
oxed{
	ilde	au_r(x,y)
=
rac{d_{mathrm{hav}}(x,y)}{v_r}.
}
]

### Important

Haversine is only a **cheap ranking approximation**.

It is never used for final route execution.

The shortlisted robots are still evaluated with exact graph-distance and battery-feasible routing.

---

## 16. Multiple orders per robot

A robot may already have several active orders.

Therefore, for a newly arrived order (j), we must consider different feasible placements of its:

- pickup;
- drop-off;

inside the robot's existing stop sequence.

For example, if a robot already serves (A) and (B), feasible candidates may produce sequences such as:

- (A ightarrow B ightarrow j);
- (A ightarrow j ightarrow B);
- (j ightarrow A ightarrow B);

subject to:

- pickup before drop-off for every order;
- payload/volume capacity at every stage;
- already-picked-up items cannot be picked up again;
- route feasibility.

---

## 17. Cheap cumulative-loss score

Let:

- (r) = candidate robot;
- (j) = new order;
- (O_r) = all active orders already assigned to robot (r);
- (S_r) = current stop sequence;
- (q) = one feasible insertion of the new pickup and drop-off;
- (S_r^q) = stop sequence after insertion (q);
- (hat T_o(S)) = estimated request-to-delivery time of order (o) under sequence (S), using Haversine travel estimates and approximate charging time.

Current estimated loss:

[
oxed{
hat L_{mathrm{before}}(r)
=
sum_{oin O_r}
L(
hat T_o(S_r),
D_o,
w_o
).
}
]

Estimated loss after insertion (q):

[
oxed{
hat L_{mathrm{after}}(r,j,q)
=
sum_{oin O_rcup{j}}
L(
hat T_o(S_r^q),
D_o,
w_o
).
}
]

Estimated incremental loss:

[
oxed{
widehat{Delta L}(r,j,q)
=
hat L_{mathrm{after}}(r,j,q)
-
hat L_{mathrm{before}}(r).
}
]

Robot heuristic score:

[
oxed{
H_{mathrm{assign}}(r,j)
=
min_{qin Q_r(j)}
widehat{Delta L}(r,j,q).
}
]

where (Q_r(j)) is the set of feasible insertion sequences for the new order.

**Lower is better.**

---

## 18. Why total loss matters

We do not minimize only the loss or delivery time of the newly arrived order.

Example:

Robot (R_1):

- can deliver the new order quickly;
- but doing so delays an existing high-importance order;
- the existing order becomes late and receives a large penalty.

Robot (R_2):

- delivers the new order slightly later;
- but does not damage another important order.

If

[
Delta L_{R_2}<Delta L_{R_1},
]

we choose (R_2).

Thus the heuristic objective is:

[
oxed{
	ext{minimum change in cumulative loss of all affected orders}
}
]

not:

[
	ext{minimum distance}
]

and not:

[
	ext{minimum new-order delivery time}.
]

---

## 19. Approximate charging in the cheap stage

The cheap stage should consider battery effects without performing the full exact battery-routing search.

Current approximation:

- estimate energy consumption from Haversine route length;
- compare cumulative estimated energy need against current battery;
- convert estimated energy deficit to charging duration using charger power.

The cheap stage does not need to reproduce:

- exact charging-station detours;
- exact charger queue behavior;
- exact battery-feasible graph route.

Those are handled in the exact second stage.

---

# Part D — Top-(K) pruning and exact selection

## 20. Top-(K) shortlist

For a new order (j):

[
	ext{feasible robots}
ightarrow
H_{mathrm{assign}}
ightarrow
	ext{sort}
ightarrow
	ext{best }K.
]

Only the best (K) robots are passed to the expensive exact search.

A percentage-based shortlist can also be tested later:

[
K
=
max(
K_{min},
lceil ho N_{mathrm{eligible}}ceil
).
]

---

## 21. Exact greedy search inside the shortlist

The final decision is exact **within the shortlist**.

For every shortlisted robot:

1. enumerate all precedence-feasible pickup/drop-off insertions;
2. enforce cumulative payload/volume feasibility;
3. use exact graph-distance routing;
4. use exact battery-feasible routing;
5. include charging;
6. calculate delivery times for every affected active order;
7. compute total exact loss after insertion;
8. compare it with the robot's exact baseline loss.

For robot (r) and insertion (q):

[
oxed{
Delta L_{mathrm{exact}}(r,j,q)
=
sum_{oin O_rcup{j}}
L_o^{after}
-
sum_{oin O_r}
L_o^{before}.
}
]

Then:

[
oxed{
(r^*,q^*)
=
argmin_{
rin C_K,;
qin Q_r(j)
}
Delta L_{mathrm{exact}}(r,j,q).
}
]

Variables:

- (C_K) = top-(K) heuristic robot shortlist;
- (Q_r(j)) = feasible insertion sequences for order (j) on robot (r).

This is a greedy decision with respect to **all orders currently known at the decision epoch**.

---

# Part E — Future anticipatory extension

## 22. Current version

The current assignment objective considers:

[
oxed{
	ext{loss of all currently known orders}.
}
]

It does not yet simulate or optimize hypothetical future arrivals during the final insertion decision.

---

## 23. Later predictive objective

Later, we may extend the final objective to:

[
oxed{
Delta L_{mathrm{current}}(r,j)
+
gamma
E[
L_{mathrm{future}}
mid
	ext{state after assignment},
hatlambda
].
}
]

The posterior spatial demand model can provide the future-arrival distribution.

This extension is deliberately deferred until the current greedy system is implemented and evaluated cleanly.

---

# Part F — Experimental plan

## 24. Shortlist tuning

On smaller instances where exhaustive robot evaluation is practical, full exact search serves as ground truth.

For different (K) values, measure:

### Recall@K

[
oxed{
operatorname{Recall@K}
=
P(
	ext{globally best exact robot is contained in top-}K
).
}
]

### Regret

[
oxed{
operatorname{Regret}
=
L_{mathrm{shortlist}}
-
L_{mathrm{full exact search}}.
}
]

Also measure:

- decision runtime;
- number of exact robot evaluations;
- number of exact insertion evaluations;
- final simulation objective.

The desired operating point is a large runtime reduction with negligible loss degradation.

---

## 25. Ablation experiments

Planned comparisons include:

- no reservation vs reservation;
- global/homogeneous reservation vs robot-type-specific reservation;
- recent-demand reservation selection vs posterior spatial-demand selection;
- uniform cluster prior vs size-aware prior;
- HDBSCAN regions vs alternative full-partition clustering;
- full exact robot evaluation vs Haversine top-(K);
- different shortlist sizes;
- different workloads;
- different fleet heterogeneity;
- different graphs;
- different priority distributions;
- different demand distributions;
- different travel-time stochasticity.

---

## 26. Benchmark policies

The project should compare the proposed algorithm against several reference policies:

1. Nearest-Available Robot (NAR);
2. Myopic A — earliest feasible completion;
3. Myopic B — availability + battery aware;
4. Reactive Insertion (RI);
5. Reassignment Policy (RP);
6. adapted anticipatory policy from the literature;
7. proposed policy;
8. optional offline oracle on small instances.

---

# Part G — Simulation-time convention

## 27. Main experiments

The main simulator is event-driven.

At simulated time (t):

1. an event occurs;
2. the policy computes its decision;
3. simulated time remains frozen during computation;
4. wall-clock computation time is measured separately.

This makes the primary benchmark reproducible and independent of the machine used to run it.

---

## 28. Later latency-aware realism experiment

Later, test whether the frozen-time assumption matters:

1. planning starts at simulation time (t);
2. robots continue their already committed motion while the planner computes;
3. measured planner wall-clock latency is converted into simulation time;
4. the selected action is applied to the state reached after that delay.

Compare the final objective under:

- frozen planning time;
- latency-aware planning time.

This will quantify whether computation delay is negligible in the intended operating regime.

---

# Part H — Implementation status

## Implemented / currently present on the design branch

- heterogeneous robot simulator;
- multi-order robot schedules;
- exact pickup/drop-off insertion machinery;
- exact cumulative incremental-loss benchmark;
- battery-aware routing and charging;
- charging queues;
- interruptible busy-robot routing;
- HDBSCAN clustering infrastructure;
- complete node-to-cluster coverage;
- persisted `in_cluster` node attribute;
- size-aware Gamma-Poisson demand model;
- posterior demand aggregation for importance thresholds;
- demand-weighted reservation-score infrastructure;
- reusable Haversine distance calculation;
- Haversine-based approximate all-order loss shortlist runner.

## Deferred

- FCNN generation and training;
- final FCNN-driven reservation wiring;
- tuning (K);
- full ablation suite;
- predictive future-order loss;
- latency-aware simulation.

---

# Final proposed decision pipeline

[
oxed{
egin{aligned}
&	ext{Offline HDBSCAN partition}\
&downarrow\
&	ext{Online Gamma-Poisson posterior } hatlambda_{z,c}\
&downarrow\
&	ext{FCNN predicts reservation fractions } alpha_{k,c}\
&downarrow\
&	ext{Choose concrete reserved robots using }H_{mathrm{reserve}}\
&downarrow\
&	ext{New order arrives}\
&downarrow\
&	ext{Hard feasibility filtering}\
&downarrow\
&	ext{Haversine-based all-order }H_{mathrm{assign}}\
&downarrow\
&	ext{Top-}K	ext{ robots}\
&downarrow\
&	ext{Exact graph + battery + all-insertion search}\
&downarrow\
&	ext{Choose minimum exact cumulative }Delta L
end{aligned}
}
]

The central computational idea is:

> **Use Haversine distance to cheaply identify promising robots, but preserve the true project objective by making both the heuristic score and the final exact decision depend on the cumulative loss of every order affected by the robot's route.**
