import math
from typing import Optional, cast

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import (
    StateVector2D,
    TimeDuration,
    TimePoint,
)
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.geometry.transform import rotate, translate
from nuplan.planning.simulation.history.simulation_history_buffer import (
    SimulationHistoryBuffer,
)
from nuplan.planning.simulation.observation.abstract_observation import (
    AbstractObservation,
)
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)
from nuplan.planning.simulation.trajectory.predicted_trajectory import (
    PredictedTrajectory,
    WaypointTypes,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from typing_extensions import override

from arbitration_pdm.common.utils.geometry import rotation_matrix_2d


class ConstantVelocityAgents(AbstractObservation):
    """
    Simulate agents based on constant velocity.
    """

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        radius: float = 100,
    ):
        """
        Constructor for ConstantVelocityAgents.

        :param trajectory_sampling: The sampling parameters for the predicted trajectory.
        :param radius: [m] Only agents within this radius around the ego will be simulated.
        """
        self.trajectory_sampling: TrajectorySampling = trajectory_sampling
        self._radius: float = radius

        self.current_iteration: int = 0
        self.agent_states: Optional[DetectionsTracks] = None

    @override
    def reset(self) -> None:
        self.current_iteration = 0

    @override
    def observation_type(self) -> type[Observation]:
        return DetectionsTracks

    @override
    def initialize(self) -> None:
        pass

    @override
    def get_observation(self) -> DetectionsTracks:
        """
        Returns the current observation of the agents.
        :return: DetectionsTracks containing the predicted trajectories of agents.
        """
        if self.agent_states is None:
            raise ValueError(
                "Agent states have not been initialized. Call update_observation first."
            )

        return self.agent_states

    @override
    def update_observation(
        self,
        iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        history: SimulationHistoryBuffer,
    ) -> None:
        self.current_iteration = next_iteration.index

        ego_state = history.current_state[0]
        self.agent_states = cast(DetectionsTracks, history.current_state[1])

        for agent in self.agent_states.tracked_objects.get_agents():
            # Skip agents that are too far away from the ego vehicle
            if ego_state.center.distance_to(agent.box.center) > self._radius:
                continue

            # Predict the next state based on constant velocity
            agent.predictions = [
                self._predict_constant_velocity(
                    agent,
                    next_iteration.time_point,
                )
            ]

    def _predict_constant_velocity(
        self,
        agent: Agent,
        t0: TimePoint,
    ) -> PredictedTrajectory:
        """
        Predict the trajectory of an agent assuming constant velocity.
        :param agent: The agent to predict the trajectory for.
        :param t0: The time point at which the prediction starts.
        :return: A PredictedTrajectory with the predicted waypoints.
        """
        box: OrientedBox = agent.box
        velocity: StateVector2D = agent.velocity
        yaw_rate = agent.angular_velocity
        if yaw_rate is None or math.isnan(yaw_rate):
            yaw_rate = 0.0

        time_point = t0
        waypoints: list[Waypoint] = []

        translate_by = self.trajectory_sampling.step_time * velocity.array
        rotate_by = self.trajectory_sampling.step_time * yaw_rate
        rotate_by_matrix = rotation_matrix_2d(rotate_by)

        # trajectory_sampling.step_time but as a TimeDuration
        delta_t = TimeDuration.from_s(self.trajectory_sampling.step_time)

        assert self.trajectory_sampling.num_poses is not None
        for _ in range(self.trajectory_sampling.num_poses):
            center = translate(box.center, translate_by)
            center = rotate(center, rotate_by_matrix)
            box = OrientedBox.from_new_pose(box, center)

            waypoints.append(Waypoint(time_point, box, velocity))
            time_point += delta_t

        return PredictedTrajectory(
            probability=1.0,
            waypoints=cast(list[Optional[WaypointTypes]], waypoints),
        )
