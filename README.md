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

## Benchmark environments

The project will combine controlled synthetic graphs/workloads with real road graphs and historical or semi-real demand traces.

## Simulation time vs. planning compute time

The simulator is event-driven. When an event at simulated time `t` requires a planning/assignment decision, the simulated clock is held at `t` until the policy returns its decision. Wall-clock computation time is measured separately and does **not** make robots move forward in simulated time.

This convention is intentional:

- **Hardware-independent comparisons.** Our main experimental question is whether one planning policy makes better routing/assignment decisions than another. If CPU time advanced the simulated world, the same algorithm could obtain a better delivery score simply by running on a faster machine, which would mix algorithm quality with hardware speed.
- **Reproducibility.** Freezing simulated time gives the same simulated trajectory when the same deterministic policy/scenario is run on a laptop, a CI runner, or a more powerful server.
- **Deployment compute can be provisioned independently.** A real fleet operator can use substantially more/parallel compute than the shared machine used for our experiments. Route decisions are also made on the scale of robot trips that typically last many minutes, so sub-second planning latency would usually be negligible relative to physical travel time.
- **Planning latency is still reported.** We record wall-clock runtime and decision/search statistics rather than pretending computation is free. If planning latency is large enough to be operationally important, a separate realism experiment can explicitly inject measured decision delay: robots would continue their already-committed motion while the planner computes, and the chosen action would be applied to the state reached when computation finishes.

Therefore benchmark objective values represent the quality of the planning policy under the standard instantaneous-decision simulation convention, while computational runtime is evaluated as a separate practicality metric.

## Status

The simulator currently supports heterogeneous robots, battery-aware routing, charging stations and queues, interruptible return-to-charge behavior, busy-robot scheduling, and reactive pickup/dropoff insertion. Benchmark and runtime optimization work is ongoing.
