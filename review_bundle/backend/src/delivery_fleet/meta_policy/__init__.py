"""Offline counterfactual training for a time-window policy selector.

Importing this package never starts workers, loads maps, or runs training.
"""

# Every policy the shared runtime can execute (kept for parity tests and ablations).
POLICIES = (
    "myopic_ab", "reactive_insertion", "paper_sa_adapted", "full_no_nn",
    "full_no_idle", "full", "full_queue_aware", "full_coordinated_idle",
    "queue_mle_reservation", "full_mle_reservation",
)

# Default candidate portfolio, chosen from 35 completed test scenarios in the 9
# training cities (v2-v7 campaigns). Two families with complementary regimes:
#   no idle control, not queue-aware: full_no_idle, paper_sa_adapted
#     (best in Paris, Barcelona, Johannesburg; up to 4.6x worse when chargers congest)
#   queue-aware, legacy or coordinated idle, with or without NN reservation:
#     full_queue_aware (NN), queue_mle_reservation (no reservation in practice),
#     full_coordinated_idle (NN), full_mle_reservation (no reservation in practice)
#     (never worse than 1.2x the best; best in Moscow, NYC and the small cities)
# MLE reservation never activated in any of 72 campaign runs, so the two MLE
# policies are the no-reservation twins of the two NN queue-aware policies.
# full_mle_reservation is also the best single fixed policy (1.044x).
# Per-scenario oracle over these six equals the best of all eight candidates;
# best policy per city in hindsight: 1.006x.
# Excluded: myopic_ab (1.24-17x worse), reactive_insertion (never best, 8x slower
# on big maps), full and full_no_nn (dominated).
DEFAULT_PORTFOLIO = ("paper_sa_adapted", "full_no_idle", "full_queue_aware",
                     "full_coordinated_idle", "queue_mle_reservation", "full_mle_reservation")

# Too expensive to label (one episode: Beijing >=1100 h, Sydney ~290 h CPU).
DEFAULT_EXCLUDED_CITIES = ("beijing", "sydney")
