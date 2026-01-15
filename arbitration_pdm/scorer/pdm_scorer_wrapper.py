from typing import Optional, Union, final

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map import RoadBlockGraphEdgeMapObject
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath

import arbitration_pdm.scorer.utils as scorer_utils
from arbitration_pdm.scorer.pdm_scorer import (
    PDMScorer,
)


@final
class PDMTrajectoryScorer:
    """Wrapper scorer that mirrors planner lifecycle and reuses PDMScorer.

    Usage:
    - Construct with `PlannerInitialization` and `proposal_sampling`.
    - Call `update(current_input: PlannerInput)` each planning iteration to refresh
      `PDMObservation`, drivable map and centerline.
    - Call `score(trajectory | list[trajectory]) -> np.ndarray` to obtain scores.

    This class will raise if route information / centerline cannot be produced (strict mode).
    """

    def __init__(
        self,
        initialization: PlannerInitialization,
        proposal_sampling: TrajectorySampling,
        map_radius: Optional[float] = None,
    ) -> None:
        """Create the wrapper and copy route dicts from initialization.

        :param initialization: PlannerInitialization providing `map_api` and route ids.
        :param proposal_sampling: proposal sampling used for scoring (must match trajectories).
        :param map_radius: optional override for drivable area radius.
        """
        self._proposal_sampling = proposal_sampling
        self._map_api = initialization.map_api
        self._map_radius = map_radius if map_radius is not None else 50

        # route dicts (copied from planner._load_route_dicts)
        self._route_roadblock_dict: dict[str, RoadBlockGraphEdgeMapObject] = {}
        self._route_lane_dict: dict[str, LaneGraphEdgeMapObject] = {}
        self._load_route_dicts(initialization.route_roadblock_ids)

        # observation and scorer
        # for observation we need trajectory_sampling and proposal_sampling; for simplicity
        # use the same sampling object for both since caller promised identical sampling.
        self._observation = PDMObservation(
            proposal_sampling,
            proposal_sampling,
            self._map_radius,
        )
        self._scorer = PDMScorer(proposal_sampling)

        # simulator: always simulate provided trajectories before scoring
        self._simulator = PDMSimulator(proposal_sampling)

        # dynamic state
        self._initial_ego_state: Optional[EgoState] = None
        self._drivable_area_map = None
        self._centerline: Optional[PDMPath] = None

    def _load_route_dicts(self, route_roadblock_ids: list[str]) -> None:
        """
        Loads roadblock and lane dictionaries of the target route from the map-api.
        :param route_roadblock_ids: ID's of on-route roadblocks
        """
        # remove repeated ids while remaining order in list
        route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))

        self._route_roadblock_dict = {}
        self._route_lane_dict = {}

        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(
                id_, SemanticMapLayer.ROADBLOCK_CONNECTOR
            )

            self._route_roadblock_dict[block.id] = block

            for lane in block.interior_edges:
                self._route_lane_dict[lane.id] = lane

    def update(self, current_input: PlannerInput) -> None:
        """Update internal observation and environment using PlannerInput.

        Mirrors planner behavior: updates `PDMObservation`, drivable area map and computes centerline.
        """
        ego_state, observation = current_input.history.current_state
        self._initial_ego_state = ego_state

        # update observation (forecasted occupancy maps)
        self._observation.update(
            ego_state,
            observation,
            current_input.traffic_light_data,
            self._route_lane_dict,
        )

        # drivable area map
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, self._map_radius
        )

        # compute centerline identical to planner
        # need a starting lane
        current_lane = scorer_utils.get_starting_lane(
            ego_state, self._drivable_area_map, self._route_lane_dict
        )
        if current_lane is None:
            raise AssertionError(
                "PDMTrajectoryScorer: could not determine starting lane for centerline"
            )

        centerline_discrete_path = scorer_utils.get_discrete_centerline(
            current_lane, self._route_lane_dict, self._route_roadblock_dict
        )
        self._centerline = PDMPath(centerline_discrete_path)

    def score(
        self, trajectories: Union[InterpolatedTrajectory, list[InterpolatedTrajectory]]
    ) -> npt.NDArray[np.float64]:
        """Score one or multiple InterpolatedTrajectory objects.

        :param trajectories: single trajectory or list of trajectories
        :return: numpy array of scores with length = n_trajectories
        """
        if isinstance(trajectories, InterpolatedTrajectory):
            trajectories_list = [trajectories]
        else:
            trajectories_list = list(trajectories)

        # Convert each trajectory to a (T, state_dim) array
        states_list = [
            scorer_utils.trajectory_to_state_array(traj, self._proposal_sampling)
            for traj in trajectories_list
        ]

        # Stack into shape (n, T, state_dim)
        states = np.stack(states_list, axis=0)

        if self._initial_ego_state is None:
            raise AssertionError(
                "PDMTrajectoryScorer: scorer.update(current_input) must be called before score()"
            )

        if self._centerline is None:
            raise AssertionError(
                "PDMTrajectoryScorer: centerline not initialized; call update() after initialization with valid route"
            )

        # simulate closed-loop execution traces starting from the real ego state
        states = self._simulator.simulate_proposals(states, self._initial_ego_state)

        assert self._drivable_area_map is not None, "drivable area map not initialized"
        scores = self._scorer.score_proposals(
            states,
            self._initial_ego_state,
            self._observation,
            self._centerline,
            self._route_lane_dict,
            self._drivable_area_map,
            self._map_api,
        )

        return scores

    def is_trajectories_valid(
        self,
        trajectories: Union[InterpolatedTrajectory, list[InterpolatedTrajectory]],
        *,
        infraction: str = "collision",
        time_to_infraction_threshold: float = 2.0,
        max_ego_speed: float = 5.0,
    ) -> npt.NDArray[np.bool_]:
        """Validate one or multiple InterpolatedTrajectory objects.

        Returns a boolean numpy array (or scalar) indicating whether each trajectory is safe.
        """
        if self._initial_ego_state is None:
            raise AssertionError(
                "PDMTrajectoryScorer: scorer.update(current_input) must be called before validation()"
            )

        if self._centerline is None:
            raise AssertionError(
                "PDMTrajectoryScorer: centerline not initialized; call update() after initialization with valid route"
            )

        if infraction not in ("collision", "ttc"):
            raise AssertionError("infraction must be 'collision' or 'ttc'")

        if isinstance(trajectories, InterpolatedTrajectory):
            traj_list = [trajectories]
        else:
            traj_list = list(trajectories)

        # Convert each trajectory to state array using the helper
        states_list = [
            scorer_utils.trajectory_to_state_array(traj, self._proposal_sampling)
            for traj in traj_list
        ]

        # Stack into shape (n, T, state_dim)
        states = np.stack(states_list, axis=0)

        # simulate provided trajectories so infraction times reflect realized execution
        states = self._simulator.simulate_proposals(states, self._initial_ego_state)

        # call score_proposals which populates time-to-infraction indices
        _ = self._scorer.score_proposals(
            states,
            self._initial_ego_state,
            self._observation,
            self._centerline,
            self._route_lane_dict,
            self._drivable_area_map,
            self._map_api,
        )

        results = np.ones(len(traj_list), dtype=bool)
        ego_speed = self._initial_ego_state.dynamic_car_state.speed

        for i in range(len(traj_list)):
            if infraction == "ttc":
                t = self._scorer.time_to_ttc_infraction(i)
            else:
                t = self._scorer.time_to_at_fault_collision(i)

            results[i] = not (
                t <= time_to_infraction_threshold and ego_speed <= max_ego_speed
            )

        return results[0] if len(results) == 1 else results

    def is_trajectory_valid(self, trajectory: InterpolatedTrajectory, **kwargs) -> bool:
        """Convenience wrapper returning a single boolean for a single trajectory."""
        return bool(self.is_trajectories_valid(trajectory, **kwargs))
