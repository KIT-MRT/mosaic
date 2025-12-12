from datetime import timedelta
from typing import List, Optional

from nuplan.common.actor_state.state_representation import (
    StateSE2,
)
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData
from nuplan.planning.simulation.history.simulation_history_buffer import (
    SimulationHistoryBuffer,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)

from arbitration_pdm.common.types import SurroundingObject, VehicleState
from arbitration_pdm.common.utils.time_conversion import to_timedelta


class EnvironmentModel:
    def __init__(self) -> None:
        self._route_roadblock_ids: Optional[list[str]] = None
        self._mission_goal: Optional[StateSE2] = None
        self._map_api: Optional[AbstractMap] = None

        self._iteration: Optional[SimulationIteration] = None
        self._history: Optional[SimulationHistoryBuffer] = None
        self._traffic_light_data: Optional[list[TrafficLightStatusData]] = None

    def initialize(self, planner_initialization: PlannerInitialization) -> None:
        self._route_roadblock_ids = planner_initialization.route_roadblock_ids
        self._mission_goal = planner_initialization.mission_goal
        self._map_api = planner_initialization.map_api

    def update(self, planner_input: PlannerInput) -> None:
        self._iteration = planner_input.iteration
        self._history = planner_input.history
        self._traffic_light_data = planner_input.traffic_light_data

    @property
    def planner_initialization(self) -> PlannerInitialization:
        """
        :return: The PlannerInitialization for the current iteration.
        """
        if (
            self._map_api is None
            or self._mission_goal is None
            or self._route_roadblock_ids is None
        ):
            raise ValueError(
                "EnvironmentModel has not been initialized. Call initialize() first."
            )
        return PlannerInitialization(
            map_api=self._map_api,
            mission_goal=self._mission_goal,
            route_roadblock_ids=self._route_roadblock_ids,
        )

    @property
    def planner_input(self) -> PlannerInput:
        """
        :return: The PlannerInput for the current iteration.
        """
        if self._iteration is None or self._history is None:
            raise ValueError(
                "EnvironmentModel has not fully initialized. Call initialize() and update() first."
            )
        return PlannerInput(
            iteration=self._iteration,
            history=self._history,
            traffic_light_data=self._traffic_light_data,
        )

    @property
    def current_time_delta(self) -> timedelta:
        """
        :return: The current time point in the simulation as a timedelta.
        """
        if self._iteration is None:
            raise ValueError(
                "EnvironmentModel has not fully initialized. Call initialize() and update() first."
            )
        return to_timedelta(self._iteration.time_point)

    @property
    def map_api(self) -> AbstractMap:
        if self._map_api is None:
            raise ValueError("Map API has not been initialized.")
        return self._map_api

    @property
    def ego_state(self):
        """
        :return: The ego state at the current time point.
        """
        if self._history is None or not self._history.ego_states:
            raise ValueError("History has not been initialized or is empty.")
        return self._history.ego_states[-1]

    @property
    def custom_vehicle_state(self) -> VehicleState:
        """
        TODO: Utilize nuPlan's VehicleState directly
        TODO: Error handling instead of silent return
        """

        if self._history is None or not self._history.ego_states:
            # Default fallback
            return VehicleState(x=0.0, y=0.0, heading=0.0)

        current_ego = self._history.ego_states[-1]

        # Extract velocity if available
        velocity = getattr(
            current_ego.dynamic_car_state.rear_axle_velocity_2d,
            "magnitude",
            lambda: 0.0,
        )()

        return VehicleState(
            x=current_ego.rear_axle.x,
            y=current_ego.rear_axle.y,
            heading=current_ego.rear_axle.heading,
            velocity=velocity,
        )

    @property
    def custom_objects(self) -> List[SurroundingObject]:
        """
        TODO: Utilize builtin nuPlan type
        TODO: Error handling instead of silent return
        """
        surrounding_objects: List[SurroundingObject] = []

        if self._history is None or not self._history.observations:
            return surrounding_objects

        latest_observation = self._history.observations[-1]

        if not hasattr(latest_observation, "tracked_objects"):
            return surrounding_objects

        for tracked_obj in latest_observation.tracked_objects.tracked_objects:
            surrounding_objects.append(
                SurroundingObject(
                    x=tracked_obj.center.x,
                    y=tracked_obj.center.y,
                    heading=tracked_obj.center.heading,
                    length=tracked_obj.box.length,
                    width=tracked_obj.box.width,
                    object_type=getattr(
                        tracked_obj.tracked_object_type, "name", "vehicle"
                    ).lower(),
                )
            )

        return surrounding_objects
