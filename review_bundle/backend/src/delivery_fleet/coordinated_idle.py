"""Opt-in coordination for editable idle routes, including committed-edge delay."""

from dataclasses import replace
from typing import Mapping, Sequence

from .idle_planning import ChargingCalendar, IdleAction, IdleFleetPlanner, IdleRobotSnapshot


class CoordinatedIdleFleetPlanner(IdleFleetPlanner):
    def _candidate_actions(
        self,
        robot: IdleRobotSnapshot,
        ranked_clusters: Sequence[int],
        cluster_value: Mapping[int, float],
        now_min: float,
        calendar: ChargingCalendar,
    ) -> list[IdleAction]:
        # A moving idle robot can change only the path after its next node.
        # Its snapshot already removes that edge's energy; don't count it twice.
        delay = max(0.0, robot.available_in_min)
        actions = super()._candidate_actions(
            robot, ranked_clusters, cluster_value, now_min + delay, calendar
        )
        return [replace(action, ready_min=action.ready_min + delay)
                for action in actions]
