from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


class Command:
    def __init__(self, name: str, trajectory: AbstractTrajectory) -> None:
        self.name: str = name
        self.trajectory: AbstractTrajectory = trajectory
