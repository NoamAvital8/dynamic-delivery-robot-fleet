"""Project-wide V1 defaults shared across the simulator.

The charging-network coverage radius and minimum full-battery robot range are
paired deliberately: every graph node is at most 2 km from a charger, while
every robot must be able to travel at least 4 km on a full battery. Therefore,
a fully charged robot can travel from a charger to any node in that charger's
coverage region and still have enough nominal range to return to a charger.
"""

MAX_DISTANCE_TO_CHARGING_STATION_M = 2_000.0
MIN_ROBOT_FULL_BATTERY_RANGE_M = 4_000.0

# V1 benchmark execution defaults.
PICKUP_SERVICE_TIME_MIN = 1.0
DROPOFF_SERVICE_TIME_MIN = 1.0
DETERMINISTIC_TRAVEL_TIMES = True
PENDING_ORDER_DISCIPLINE = "fifo"
CONTINUE_UNTIL_ALL_RELEASED_ORDERS_FINISH = True
LATE_ORDERS_ARE_STILL_DELIVERED = True

if 2 * MAX_DISTANCE_TO_CHARGING_STATION_M > MIN_ROBOT_FULL_BATTERY_RANGE_M:
    raise RuntimeError(
        "charging coverage requires at least twice the station coverage radius "
        "as minimum full-battery robot range"
    )
