from dataclasses import dataclass

from arbitration_graphs.typing import Time
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)


class Command:
    def __init__(self, name: str, trajectory: InterpolatedTrajectory) -> None:
        self.name: str = name
        self.trajectory: InterpolatedTrajectory = trajectory


@dataclass
class StampedCommand:
    stamp: Time
    command: Command
