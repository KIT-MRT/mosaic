from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from nuplan.common.actor_state.state_representation import (
    StateSE2,
)
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import (
    LaneGraphEdgeMapObject,
    RoadBlockGraphEdgeMapObject,
)
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData
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
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath

from mosaic.common.utils import map as map_utils
from mosaic.common.utils.time_conversion import to_timedelta
from mosaic.scorer.pdm_scorer import PDMScorer


class EnvironmentModel:
    @dataclass
    class Parameters:
        trajectory_sampling: TrajectorySampling
        proposal_sampling: TrajectorySampling
        map_radius: float
        ttc_horizon: float = 3.0

    def __init__(self, parameters: Parameters) -> None:
        self.parameters: EnvironmentModel.Parameters = parameters

        self._route_roadblock_ids: Optional[list[str]] = None
        self._mission_goal: Optional[StateSE2] = None
        self._map_api: Optional[AbstractMap] = None

        self._route_roadblock_dict: dict[str, RoadBlockGraphEdgeMapObject] = {}
        self._route_lane_dict: dict[str, LaneGraphEdgeMapObject] = {}

        self._iteration: Optional[SimulationIteration] = None
        self._history: Optional[SimulationHistoryBuffer] = None
        self._traffic_light_data: Optional[list[TrafficLightStatusData]] = None

        self._observation: Optional[PDMObservation] = None
        self._drivable_area_map: Optional[PDMOccupancyMap] = None
        self._route_center_line: Optional[PDMPath] = None

    def initialize(self, planner_initialization: PlannerInitialization) -> None:
        self._route_roadblock_ids = planner_initialization.route_roadblock_ids
        self._mission_goal = planner_initialization.mission_goal
        self._map_api = planner_initialization.map_api

        self._load_route_dicts(planner_initialization.route_roadblock_ids)

        self._observation = PDMObservation(
            self.parameters.trajectory_sampling,
            self.parameters.proposal_sampling,
            self.parameters.map_radius,
            ttc_horizon=self.parameters.ttc_horizon,
        )

    def update(self, planner_input: PlannerInput) -> None:
        assert self._map_api is not None, "You must call initialize() first."
        assert self._observation is not None, "You must call initialize() first."

        self._iteration = planner_input.iteration
        self._history = planner_input.history
        self._traffic_light_data = planner_input.traffic_light_data

        ego_state, observation = self._history.current_state
        assert self._traffic_light_data is not None, (
            "Traffic light data is not available."
        )

        self._observation.update(
            ego_state,
            observation,
            self._traffic_light_data,
            self._route_lane_dict,
        )
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, self.parameters.map_radius
        )
        current_lane = map_utils.get_starting_lane(
            ego_state, self._drivable_area_map, self._route_lane_dict
        )
        centerline_discrete_path = map_utils.get_discrete_centerline(
            current_lane, self._route_lane_dict, self._route_roadblock_dict
        )
        self._route_center_line = PDMPath(centerline_discrete_path)

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
    def route_lane_dict(self) -> dict[str, LaneGraphEdgeMapObject]:
        """
        :return: The dictionary of on-route lane ID's to LaneGraphEdgeMapObjects.
        """
        return self._route_lane_dict

    @property
    def observation(self) -> PDMObservation:
        """
        :return: The PDMObservation at the current time point.
        """
        if self._observation is None:
            raise ValueError(
                "Observation has not been initialized. Call update() after initialize()."
            )
        return self._observation

    @property
    def drivable_area_map(self) -> PDMOccupancyMap:
        """
        :return: The drivable area map at the current time point.
        """
        if self._drivable_area_map is None:
            raise ValueError(
                "Drivable area map has not been initialized. Call update() after initialize()."
            )
        return self._drivable_area_map

    @property
    def route_center_line(self) -> PDMPath:
        """
        :return: The center line of the current route as a PDMPath.
        """
        if self._route_center_line is None:
            raise ValueError(
                "Centerline has not been initialized. Call update() after initialize()."
            )
        return self._route_center_line

    def _load_route_dicts(self, route_roadblock_ids: list[str]) -> None:
        """
        Loads roadblock and lane dictionaries of the target route from the map-api.
        :param route_roadblock_ids: ID's of on-route roadblocks
        """
        # remove repeated ids while remaining order in list
        route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))

        self._route_roadblock_dict = {}
        self._route_lane_dict = {}

        assert self._map_api is not None, "Map API has not been initialized."
        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(
                id_, SemanticMapLayer.ROADBLOCK_CONNECTOR
            )

            self._route_roadblock_dict[block.id] = block

            for lane in block.interior_edges:
                self._route_lane_dict[lane.id] = lane
