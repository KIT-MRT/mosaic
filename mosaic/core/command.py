from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from arbitration_graphs.typing import Time
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


class Command:
    def __init__(
        self,
        name: str,
        trajectory: AbstractTrajectory,
        simulated_states: npt.NDArray[np.float64],
    ) -> None:
        self.name: str = name
        self.trajectory: AbstractTrajectory = trajectory
        self.simulated_states: npt.NDArray[np.float64] = simulated_states


@dataclass
class StampedCommand:
    stamp: Time
    command: Command
