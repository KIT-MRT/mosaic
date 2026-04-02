from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy.typing as npt

from mosaic.core.environment_model import EnvironmentModel
from mosaic.scorer.scoring_input import ScoringInput


@dataclass
class MetricResult:
    scores: npt.NDArray  # (n_proposals,) in [0,1]
    metadata: dict[str, Any] = field(default_factory=dict)


class AbstractMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def compute(
        self, scoring_input: ScoringInput, environment_model: EnvironmentModel
    ) -> MetricResult: ...


class MultiplicativeMetric(AbstractMetric):
    """Marker for safety-gate metrics whose scores are multiplied together."""

    pass


class WeightedMetric(AbstractMetric):
    """Metric whose score contributes to a weighted average."""

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def weight(self) -> float:
        return self._weight
