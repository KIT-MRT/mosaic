import json
import math
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Optional, cast

from arbitration_graphs.typing import Time
from nuplan.common.actor_state.state_representation import (
    Point2D,
    TimeDuration,
    TimePoint,
)
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import (
    Intersection,
    Lane,
    LaneConnector,
    PolygonMapObject,
    PolylineMapObject,
    StopLine,
)
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely.geometry import LineString, Polygon

from mosaic.core.command import Command
from mosaic.core.environment_model import EnvironmentModel

_MAP_LAYERS: list[SemanticMapLayer] = [
    SemanticMapLayer.LANE,
    SemanticMapLayer.LANE_CONNECTOR,
    SemanticMapLayer.CROSSWALK,
    SemanticMapLayer.STOP_LINE,
    SemanticMapLayer.INTERSECTION,
]


def _polygon_to_list(polygon: Polygon) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in polygon.exterior.coords]


def _linestring_to_list(linestring: LineString) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in linestring.coords]


def _trajectory_to_waypoints(
    trajectory: AbstractTrajectory, sampling: TrajectorySampling
) -> list[dict[str, float]]:
    """Sample a trajectory uniformly and return plain-float waypoint dicts."""
    step_time = TimeDuration.from_s(sampling.step_time)
    num_poses = sampling.num_poses
    assert num_poses is not None

    time_points: list[TimePoint] = [
        trajectory.start_time + step_time * i for i in range(num_poses + 1)
    ]
    last = time_points[-1]
    end = trajectory.end_time
    if last.time_us > end.time_us and last.time_us - end.time_us <= 1:
        time_points[-1] = end

    ego_states = trajectory.get_state_at_times(time_points)

    start_us = trajectory.start_time.time_us
    waypoints: list[dict[str, float]] = []
    for tp, state in zip(time_points, ego_states):
        waypoints.append(
            {
                "t": (tp.time_us - start_us) / 1e6,
                "x": float(state.center.x),
                "y": float(state.center.y),
                "heading": float(state.center.heading),
                "v": float(state.dynamic_car_state.speed),
            }
        )
    return waypoints


class SceneRecorder:
    @dataclass
    class Parameters:
        proposal_sampling: TrajectorySampling
        map_radius_buffer_m: float = 30.0

    def __init__(self, parameters: Parameters) -> None:
        self.parameters: SceneRecorder.Parameters = parameters

        self._proposals_buffer: list[dict[str, Any]] = []
        self._observations_buffer: list[dict[str, Any]] = []
        self._ego_trail: list[tuple[float, float]] = []

        self._map_api: Optional[AbstractMap] = None
        self._ego_dimensions: Optional[dict[str, float]] = None

    def initialize_scenario(self, map_api: AbstractMap) -> None:
        self._map_api = map_api
        self._ego_dimensions = None
        self._ego_trail = []
        self._proposals_buffer = []
        self._observations_buffer = []

    def record_step(
        self,
        time: Time,
        environment_model: EnvironmentModel,
        commands: list[Command],
        selected_command_name: str,
    ) -> None:
        assert isinstance(time, timedelta)
        t_s = time.total_seconds()

        if self._ego_dimensions is None:
            footprint = environment_model.ego_state.car_footprint
            self._ego_dimensions = {
                "length": float(footprint.length),
                "width": float(footprint.width),
            }

        ego_state = environment_model.ego_state
        ego_entry = {
            "x": float(ego_state.center.x),
            "y": float(ego_state.center.y),
            "heading": float(ego_state.center.heading),
            "v": float(ego_state.dynamic_car_state.speed),
        }
        self._ego_trail.append((ego_entry["x"], ego_entry["y"]))

        proposals: list[dict[str, Any]] = []
        for command in commands:
            proposals.append(
                {
                    "command": command.name,
                    "waypoints": _trajectory_to_waypoints(
                        command.trajectory, self.parameters.proposal_sampling
                    ),
                }
            )

        self._proposals_buffer.append(
            {
                "time": t_s,
                "ego": ego_entry,
                "selected": selected_command_name,
                "proposals": proposals,
            }
        )

        history = environment_model.planner_input.history
        _, detections = history.current_state
        tracked_objects: list[dict[str, Any]] = []
        for obj in detections.tracked_objects:
            box = obj.box
            velocity = getattr(obj, "velocity", None)
            tracked_objects.append(
                {
                    "track_token": obj.metadata.track_token,
                    "type": obj.tracked_object_type.name,
                    "x": float(box.center.x),
                    "y": float(box.center.y),
                    "heading": float(box.center.heading),
                    "vx": float(velocity.x) if velocity is not None else 0.0,
                    "vy": float(velocity.y) if velocity is not None else 0.0,
                    "length": float(box.length),
                    "width": float(box.width),
                }
            )

        self._observations_buffer.append(
            {"time": t_s, "tracked_objects": tracked_objects}
        )

    def flush_logs(self, log_dir: str, scenario_name: str) -> None:
        if not self._ego_trail:
            return

        static = self._build_static_dump(scenario_name)
        with open(
            os.path.join(log_dir, f"{scenario_name}_render_static.json"), "w"
        ) as f:
            json.dump(static, f)

        with open(os.path.join(log_dir, f"{scenario_name}_proposals.jsonl"), "w") as f:
            for entry in self._proposals_buffer:
                _ = f.write(json.dumps(entry) + "\n")

        with open(
            os.path.join(log_dir, f"{scenario_name}_observations.jsonl"), "w"
        ) as f:
            for entry in self._observations_buffer:
                _ = f.write(json.dumps(entry) + "\n")

    def _build_static_dump(self, scenario_name: str) -> dict[str, Any]:
        assert self._map_api is not None

        xs = [p[0] for p in self._ego_trail]
        ys = [p[1] for p in self._ego_trail]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

        max_distance = max(
            (math.hypot(x - cx, y - cy) for x, y in self._ego_trail),
            default=0.0,
        )
        radius = max_distance + self.parameters.map_radius_buffer_m

        objects_by_layer = self._map_api.get_proximal_map_objects(
            Point2D(cx, cy), radius, _MAP_LAYERS
        )

        map_features: dict[str, list[list[list[float]]]] = {
            "lane_polygons": [],
            "lane_centerlines": [],
            "lane_left_boundaries": [],
            "lane_right_boundaries": [],
            "lane_connector_polygons": [],
            "lane_connector_centerlines": [],
            "lane_connector_left_boundaries": [],
            "lane_connector_right_boundaries": [],
            "crosswalks": [],
            "stop_lines": [],
            "intersections": [],
        }

        for layer, objs in objects_by_layer.items():
            if layer == SemanticMapLayer.LANE:
                for obj in cast(list[Lane], objs):
                    map_features["lane_polygons"].append(_polygon_to_list(obj.polygon))
                    map_features["lane_centerlines"].append(
                        _linestring_to_list(obj.baseline_path.linestring)
                    )
                    map_features["lane_left_boundaries"].append(
                        _linestring_to_list(obj.left_boundary.linestring)
                    )
                    map_features["lane_right_boundaries"].append(
                        _linestring_to_list(obj.right_boundary.linestring)
                    )
            elif layer == SemanticMapLayer.LANE_CONNECTOR:
                for obj in cast(list[LaneConnector], objs):
                    map_features["lane_connector_polygons"].append(
                        _polygon_to_list(obj.polygon)
                    )
                    map_features["lane_connector_centerlines"].append(
                        _linestring_to_list(obj.baseline_path.linestring)
                    )
                    left: Optional[PolylineMapObject] = getattr(
                        obj, "left_boundary", None
                    )
                    right: Optional[PolylineMapObject] = getattr(
                        obj, "right_boundary", None
                    )
                    if left is not None:
                        map_features["lane_connector_left_boundaries"].append(
                            _linestring_to_list(left.linestring)
                        )
                    if right is not None:
                        map_features["lane_connector_right_boundaries"].append(
                            _linestring_to_list(right.linestring)
                        )
            elif layer == SemanticMapLayer.CROSSWALK:
                for obj in cast(list[PolygonMapObject], objs):
                    map_features["crosswalks"].append(_polygon_to_list(obj.polygon))
            elif layer == SemanticMapLayer.STOP_LINE:
                for obj in cast(list[StopLine], objs):
                    map_features["stop_lines"].append(_polygon_to_list(obj.polygon))
            elif layer == SemanticMapLayer.INTERSECTION:
                for obj in cast(list[Intersection], objs):
                    map_features["intersections"].append(_polygon_to_list(obj.polygon))

        return {
            "scenario_token": scenario_name,
            "ego_dimensions": self._ego_dimensions,
            "bounding_box": [x_min, y_min, x_max, y_max],
            "map_query": {"center": [cx, cy], "radius": radius},
            "map_features": map_features,
        }
