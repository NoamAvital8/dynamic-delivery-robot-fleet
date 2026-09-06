# Dynamic Delivery Robot Fleet

Course project for **AI and Autonomous Systems**.

We study online planning and assignment for a heterogeneous fleet of autonomous delivery robots operating on a city graph under stochastic demand and travel times.

## Core setting

- Pickup-to-delivery requests arrive online.
- Robots may differ in speed, payload capacity, battery capacity, and energy consumption.
- A robot may carry multiple deliveries when feasible.
- The planner may wait for a better robot instead of assigning immediately.
- Future extensions include package transfer between robots and proactive repositioning.

## Primary objective

We minimize total importance-weighted customer waiting time:

\[
L = \sum_{r \in R} w_r \left(t_r^{delivery} - t_r^{request}\right)
\]

where `w_r` is the importance of request `r`.

If rejected/failed deliveries are allowed, an additional sufficiently large rejection penalty will be used.

## Planned benchmark policies

1. Nearest-Available Robot (NAR)
2. Myopic A — earliest feasible completion
3. Myopic B — availability + battery aware
4. Reactive Insertion (RI)
5. Reassignment Policy (RP)
6. SA-adapted / anticipatory policy
7. Proposed algorithm
8. Optional offline oracle for small instances

## Benchmark environments

The project will combine controlled synthetic graphs/workloads with real road graphs and historical or semi-real demand traces.

## Status

Repository initialized. Simulator architecture and first baseline are next.
