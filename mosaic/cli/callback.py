from __future__ import annotations

from typing import TYPE_CHECKING

from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from typing_extensions import override

if TYPE_CHECKING:
    from nuplan.planning.simulation.history.simulation_history_buffer import (
        SimulationHistorySample,
    )


class MosaicLoggingCallback(AbstractCallback):
    """Flushes buffered mosaic logs to the experiment output directory."""

    def __init__(self, output_directory: str) -> None:
        self._output_directory = str(output_directory)

    @override
    def on_initialization_start(
        self, setup: SimulationSetup, planner: AbstractPlanner
    ) -> None:
        pass

    @override
    def on_initialization_end(
        self, setup: SimulationSetup, planner: AbstractPlanner
    ) -> None:
        pass

    @override
    def on_step_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        pass

    @override
    def on_step_end(
        self,
        setup: SimulationSetup,
        planner: AbstractPlanner,
        sample: SimulationHistorySample,
    ) -> None:
        pass

    @override
    def on_planner_start(
        self, setup: SimulationSetup, planner: AbstractPlanner
    ) -> None:
        pass

    @override
    def on_planner_end(
        self,
        setup: SimulationSetup,
        planner: AbstractPlanner,
        trajectory: AbstractTrajectory,
    ) -> None:
        pass

    @override
    def on_simulation_start(self, setup: SimulationSetup) -> None:
        pass

    @override
    def on_simulation_end(
        self,
        setup: SimulationSetup,
        planner: AbstractPlanner,
        history: SimulationHistory,
    ) -> None:
        from mosaic.core.mosaic_planner import Mosaic

        if isinstance(planner, Mosaic):
            planner.flush_logs(self._output_directory, setup.scenario.scenario_name)
