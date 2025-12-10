from pathlib import Path

from arbitrator_pdm.ego_agent import EgoAgent

print("Improved normalized trajectory evaluation system loaded successfully!")
print("Configured with proper metric normalization and weighted scoring")
print("All metrics will be distributed uniformly between 0.000 and 100.000")
print("Usage: planner = EgoAgent(detailed_logging=False)")
import hydra

# Location of paths with all simulation configs
CONFIG_PATH = (
    "../nuplan-devkit/nuplan/planning/script/config/simulation"  # TODO: Hardcoded path
)
CONFIG_NAME = "default_simulation"

# Create a temporary directory to store the simulation artifacts
SAVE_DIR = "../experiments"  # TODO: Hardcoded path

# Select simulation parameters
CHALLENGE = "closed_loop_reactive_agents"  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
# OBSERVATION = 'idm_agents_observation'  # [box_observation, idm_agents_observation, lidar_pc_observation]

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
hydra.initialize(config_path=CONFIG_PATH)

# Compose the configuration
cfg = hydra.compose(
    config_name=CONFIG_NAME,
    overrides=[
        f"group={SAVE_DIR}",
        "experiment_name=planner_tutorial",
        "job_name=planner_tutorial",
        "experiment=${experiment_name}/${job_name}",
        "output_dir=${group}/${experiment}",
        f"+simulation={CHALLENGE}",
        # f'observation={OBSERVATION}',
        "scenario_filter=val14_split",
        "scenario_builder=nuplan",
        # 'worker=sequential',
        "hydra.searchpath=[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]",
    ],
)
import nest_asyncio
from nuplan.planning.script.run_simulation import run_simulation as main_simulation

nest_asyncio.apply()

planner = EgoAgent()  # 构造 planner

main_simulation(cfg, planners=planner)
simulation_folder = cfg.output_dir

# Print the simulation folder path
print(f"Simulation results are saved in: {simulation_folder}")

# Location of paths with all nuBoard configs
CONFIG_PATH = (
    "../nuplan-devkit/nuplan/planning/script/config/nuboard"  # TODO: Hardcoded path
)
CONFIG_NAME = "default_nuboard"

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
hydra.initialize(config_path=CONFIG_PATH)

# Compose the configuration
cfg = hydra.compose(
    config_name=CONFIG_NAME,
    overrides=[
        "scenario_builder=nuplan_mini",  # set the database (same as simulation) used to fetch data for visualization
        "simulation_path=../experiments/planner_tutorial/planner_tutorial",  # nuboard file path, if left empty the user can open the file inside nuBoard # TODO: Hardcoded path
    ],
)

import pandas as pd
from nuplan.planning.metrics.aggregator.weighted_average_metric_aggregator import (
    WeightedAverageMetricAggregator,
)
from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame

# Step 1: 设置仿真输出目录
output_dir = Path(
    "../experiments/planner_tutorial/planner_tutorial"
)  # TODO: Hardcoded path

# Step 2: 加载所有 .parquet 指标为 dataframe 封装。NuPlan 在跑 run_simulation() 时，会在 metrics/ 目录下输出若干 .parquet 文件。
# 每个文件代表一个 metric 的结果（比如 ego_expert_l2_error.parquet）；
# 每个文件的结构是一个 dataframe，每行是一个 scenario 的得分记录：
metrics_dir = output_dir / "metrics"

# 用 MetricStatisticsDataFrame 包装这些表格，方便后续聚合。
metric_dataframes = {}
for file in metrics_dir.glob("*.parquet"):
    df = pd.read_parquet(file)
    metric_name = file.stem
    metric_dataframes[metric_name] = MetricStatisticsDataFrame(
        metric_statistic_name=metric_name, metric_statistics_dataframe=df
    )

# Step 3: 构建 aggregator。表示你希望对所有指标赋等权（或手动设定某些指标高权重），聚合每个 scenario 的多个指标，计算评分。
aggregator = WeightedAverageMetricAggregator(
    name="default_aggregator",
    metric_weights={"default": 1.0},
    file_name="aggregator_metric.parquet",
    aggregator_save_path=output_dir / "aggregator_metric",
    multiple_metrics=[],
    challenge_name=None,
)

# Step 4: 运行聚合
aggregator(metric_dataframes)

from nuplan.planning.script.run_nuboard import main as main_nuboard

# Run nuBoard
main_nuboard(cfg)
