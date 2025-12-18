from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely.geometry import Point
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import (
    Dijkstra,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_states_to_state_array,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    normalize_angle,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


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
        require_route: bool = True,
    ) -> None:
        """Create the wrapper and copy route dicts from initialization.

        :param initialization: PlannerInitialization providing `map_api` and route ids.
        :param proposal_sampling: proposal sampling used for scoring (must match trajectories).
        :param map_radius: optional override for drivable area radius.
        :param require_route: if True, raise when route/centerline can't be computed.
        """
        self._proposal_sampling = proposal_sampling
        self._map_api = initialization.map_api
        self._map_radius = map_radius
        self._require_route = require_route

        # route dicts (copied from planner._load_route_dicts)
        self._route_roadblock_dict: Dict[str, object] = {}
        self._route_lane_dict: Dict[str, object] = {}
        route_roadblock_ids = getattr(initialization, "route_roadblock_ids", []) or []
        if route_roadblock_ids:
            self._load_route_dicts(route_roadblock_ids)
        elif require_route:
            raise AssertionError(
                "PDMTrajectoryScorer: route_roadblock_ids required for centerline computation"
            )

        # observation and scorer
        # for observation we need trajectory_sampling and proposal_sampling; for simplicity
        # use the same sampling object for both since caller promised identical sampling.
        self._observation = PDMObservation(
            proposal_sampling,
            proposal_sampling,
            map_radius if map_radius is not None else 50,
        )
        self._scorer = PDMScorer(proposal_sampling)

        # dynamic state
        self._initial_ego_state: Optional[EgoState] = None
        self._drivable_area_map = None
        self._centerline: Optional[PDMPath] = None

    def _load_route_dicts(self, route_roadblock_ids: List[str]) -> None:
        """Load roadblock and lane dictionaries of the target route from the map-api.

        Copied logic from AbstractPDMPlanner._load_route_dicts.
        """
        # remove repeated ids while preserving order
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

        # ensure map radius
        map_radius = self._map_radius if self._map_radius is not None else 50

        # update observation (forecasted occupancy maps)
        self._observation.update(
            ego_state,
            observation,
            current_input.traffic_light_data,
            self._route_lane_dict,
        )

        # drivable area map
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, map_radius
        )

        # compute centerline identical to planner
        # need a starting lane
        current_lane = self._get_starting_lane(ego_state)
        if current_lane is None:
            if self._require_route:
                raise AssertionError(
                    "PDMTrajectoryScorer: could not determine starting lane for centerline"
                )
            else:
                self._centerline = None
                return

        centerline_discrete_path = self._get_discrete_centerline(current_lane)
        self._centerline = PDMPath(centerline_discrete_path)

    def score(
        self, trajectories: Union[InterpolatedTrajectory, List[InterpolatedTrajectory]]
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
            self._trajectory_to_state_array(traj) for traj in trajectories_list
        ]

        # Stack into shape (n, T, state_dim)
        states = np.stack(states_list, axis=0)

        if self._initial_ego_state is None:
            raise AssertionError(
                "PDMTrajectoryScorer: scorer.update(current_input) must be called before score()"
            )

        if self._centerline is None and self._require_route:
            raise AssertionError(
                "PDMTrajectoryScorer: centerline not initialized; call update() after initialization with valid route"
            )

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
        trajectories: Union[InterpolatedTrajectory, List[InterpolatedTrajectory]],
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

        if self._centerline is None and self._require_route:
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
        states_list = [self._trajectory_to_state_array(traj) for traj in traj_list]

        # Stack into shape (n, T, state_dim)
        states = np.stack(states_list, axis=0)

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

    # --- helper methods copied from AbstractPDMPlanner; minimal set ---
    def _get_discrete_centerline(
        self, current_lane, search_depth: int = 30
    ) -> List[StateSE2]:
        """Compute discrete centerline by applying Dijkstra search on lane-graph.

        Logic copied from AbstractPDMPlanner._get_discrete_centerline.
        """
        roadblocks = list(self._route_roadblock_dict.values())
        roadblock_ids = list(self._route_roadblock_dict.keys())

        # find current roadblock index
        start_idx = int(
            np.argmax(np.array(roadblock_ids) == current_lane.get_roadblock_id())
        )
        roadblock_window = roadblocks[start_idx : start_idx + search_depth]

        graph_search = Dijkstra(current_lane, list(self._route_lane_dict.keys()))
        route_plan, path_found = graph_search.search(roadblock_window[-1])

        centerline_discrete_path: List[StateSE2] = []
        for lane in route_plan:
            centerline_discrete_path.extend(lane.baseline_path.discrete_path)

        return centerline_discrete_path

    def _get_starting_lane(self, ego_state: EgoState):
        """Return most suitable starting lane in ego's vicinity. Copied logic from planner."""
        starting_lane = None
        on_route_lanes, heading_error = self._get_intersecting_lanes(ego_state)

        if on_route_lanes:
            starting_lane = on_route_lanes[
                int(np.argmin(np.abs(np.array(heading_error))))
            ]
            return starting_lane

        else:
            # find any lane on-route that contains point or is closest
            closest_distance = np.inf
            for edge in self._route_lane_dict.values():
                if edge.contains_point(ego_state.center):
                    starting_lane = edge
                    break

                distance = edge.polygon.distance(ego_state.car_footprint.geometry)
                if distance < closest_distance:
                    starting_lane = edge
                    closest_distance = distance

        return starting_lane

    def _get_intersecting_lanes(self, ego_state: EgoState):
        """Return on-route lanes and heading errors where ego intersects. Copied logic."""
        assert self._drivable_area_map, (
            "PDMTrajectoryScorer: Drivable area map must be initialized first!"
        )

        ego_position_array = ego_state.rear_axle.array
        ego_rear_axle_point = Point(*ego_position_array)
        ego_heading = ego_state.rear_axle.heading

        intersecting_lanes = self._drivable_area_map.intersects(ego_rear_axle_point)

        on_route_lanes, on_route_heading_errors = [], []
        for lane_id in intersecting_lanes:
            if lane_id in self._route_lane_dict.keys():
                lane_object = self._route_lane_dict[lane_id]
                lane_discrete_path: List[StateSE2] = (
                    lane_object.baseline_path.discrete_path
                )
                lane_state_se2_array = np.array(
                    [state.array for state in lane_discrete_path], dtype=np.float64
                )

                lane_distances = (
                    ego_position_array[None, ...] - lane_state_se2_array
                ) ** 2
                lane_distances = lane_distances.sum(axis=-1) ** 0.5

                heading_error = (
                    lane_discrete_path[int(np.argmin(lane_distances))].heading
                    - ego_heading
                )
                heading_error = abs(normalize_angle(heading_error))

                on_route_lanes.append(lane_object)
                on_route_heading_errors.append(heading_error)

        return on_route_lanes, on_route_heading_errors

    def _trajectory_to_state_array(
        self, traj: InterpolatedTrajectory
    ) -> npt.NDArray[np.float64]:
        """Resample an InterpolatedTrajectory to proposal sampling and convert to state array."""
        step_time_s = self._proposal_sampling.step_time
        expected_len = self._proposal_sampling.num_poses + 1

        # Anchor sampling to the trajectory start time
        start_time = traj.start_time
        time_points = [
            type(start_time)(start_time.time_us + int(round(k * step_time_s * 1e6)))
            for k in range(expected_len)
        ]

        # Allow ≤1µs clamping at the end due to rounding artifacts
        last_tp_us = time_points[-1].time_us
        traj_end_us = traj.end_time.time_us
        if last_tp_us > traj_end_us:
            if last_tp_us - traj_end_us <= 1:
                time_points[-1] = type(start_time)(traj_end_us)
            else:
                raise AssertionError(
                    f"PDMTrajectoryScorer: trajectory time window {traj.start_time}..{traj.end_time} "
                    f"does not contain required times for proposal sampling {self._proposal_sampling}"
                )

        try:
            ego_states = traj.get_state_at_times(time_points)
        except AssertionError as exc:
            raise AssertionError(
                f"PDMTrajectoryScorer: trajectory time window {traj.start_time}..{traj.end_time} "
                f"does not contain required times for proposal sampling {self._proposal_sampling}"
            ) from exc

        if len(ego_states) != expected_len:
            raise AssertionError(
                f"PDMTrajectoryScorer: resampled trajectory length {len(ego_states)} "
                f"does not match expected {expected_len}"
            )

        return ego_states_to_state_array(ego_states)
