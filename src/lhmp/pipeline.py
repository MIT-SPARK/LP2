from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union
import pandas as pd
import numpy as np
import yaml
import pickle
import os
import logging

import spark_dsg as dsg
import plotly.graph_objects as go

from importlib import import_module

from lhmp.modules.goal.goal_module_base import GoalModuleBase
from lhmp.modules.goal.llm import LlmGoalModule
from lhmp.modules.planning.planning_module_base import PlanningModuleBase
from lhmp.modules.low_level.low_level_module_base import LowLevelModuleBase
from lhmp.dataset import LHMPDataset, load_dataset
from lhmp.utils.visualization import (
    visualize_path,
    visualize_paths_in_dsg,
    LAYER_SETTINGS,
)


class Pipeline:
    """
    Pipeline: joins all modules to produce output data
    """

    # pass class objects to constructor
    def __init__(
        self,
        GoalModuleClass: GoalModuleBase,
        PlanningModuleClass: PlanningModuleBase,
        LowLevelModuleClass: LowLevelModuleBase,
        *args,
        **kwargs,
    ):
        self.verbose = kwargs["global"]["verbose"]
        self.goal_module = GoalModuleClass(**kwargs["goal_module"])
        self.planning_module = PlanningModuleClass(**kwargs["planning_module"])
        self.low_level_module = LowLevelModuleClass(**kwargs["low_level_module"])

        if "visualization" in kwargs["global"]:
            self.visualization = kwargs["global"]["visualization"]
        else:
            self.visualization = True

    def run(
        self,
        scene_graph_path: str,
        room_labels_path: str,
        trajectory_dir: str,
        time_horizon: float,
        alpha: float = None,
        beta: float = None,
        n_past_interactions: int = None,
        n_total_interactions: int = None,
        time_after_interaction: float = None,
        future_trajectory_length: float = None,
        out_path: str = None,
        *args,
        **kwargs,
    ):
        """
        Construct dataset and run pipeline
        :param scene_graph_path: path to scene graph JSON file
        :param room_labels_path: path to room labels JSON file
        :param trajectory_dir: path to directory containing trajectory CSV files
        :param time_horizon: time horizon for trajectory prediction (past+future trajectory)
        :param alpha: fraction of time horizon to use for past trajectory
        :param beta: fraction of time horizon to use for future trajectory
        :param n_past_interactions: alternative to (alpha, beta, horizon): number of past interactions to include in past trajectory
        :param n_total_interactions: alternative to (alpha, beta, horizon): total number of interactions to include in past+future trajectory
        """

        if n_past_interactions and n_total_interactions:
            self.goal_module.depth = n_total_interactions - n_past_interactions

        dataset = load_dataset(
            scene_graph_path,
            room_labels_path,
            trajectory_dir,
            time_horizon=time_horizon,
            alpha=alpha,
            beta=beta,
            n_past_interactions=n_past_interactions,
            n_total_interactions=n_total_interactions,
            time_after_interaction=time_after_interaction,
            future_trajectory_length=future_trajectory_length,
        )

        out = {}

        self.goal_module.on_episode_init(dataset.scene_graph, dataset.room_labels)
        self.planning_module.on_episode_init(dataset.scene_graph)
        self.low_level_module.on_episode_init(dataset.scene_graph)

        if (
            "method_precompute" in kwargs["global"]
            and kwargs["global"]["method_precompute"] is not None
        ):
            method_name = kwargs["global"]["method_precompute"]
        else:
            method_name = kwargs["global"]["method"]

        for (
            trajectory_id,
            past_trajectory,
            past_trajectory_info,
            gt_future_trajectory,
            future_trajectory_info,
        ) in dataset:
            if self.verbose:
                print(f"Processing trajectory {trajectory_id}")
            if (
                "load_checkpoints" in kwargs["global"]
                and kwargs["method"] != "trajectron"
                and kwargs["global"]["load_checkpoints"]
            ):
                if os.path.exists(
                    os.path.join(
                        kwargs["output"]["folder"],
                        kwargs["global"]["scene"],
                        "out_data",
                        method_name,
                        f"data_{kwargs['global']['trajectory_part']}_{trajectory_id}.pkl",
                    )
                ):
                    with open(
                        os.path.join(
                            kwargs["output"]["folder"],
                            kwargs["global"]["scene"],
                            "out_data",
                            method_name,
                            f"data_{kwargs['global']['trajectory_part']}_{trajectory_id}.pkl",
                        ),
                        "rb",
                    ) as f:
                        out[trajectory_id] = pickle.load(f)
                    continue

            eval_timesteps = np.arange(
                gt_future_trajectory["t"].iloc[0],
                min(gt_future_trajectory["t"].iloc[-1], time_horizon),
                kwargs["global"]["eval_timestep"],
            )

            # ISP module
            if (
                "use_precomputed_interactions" in kwargs["global"]
                and kwargs["global"]["use_precomputed_interactions"]
                and os.path.exists(
                    os.path.join(
                        kwargs["output"]["folder"],
                        kwargs["global"]["scene"],
                        "out_data",
                        method_name,
                        f"data_{kwargs['global']['trajectory_part']}_{trajectory_id}.pkl",
                    )
                )
            ):
                logging.info("Using precomputed interaction tree.")
                with open(
                    os.path.join(
                        kwargs["output"]["folder"],
                        kwargs["global"]["scene"],
                        "out_data",
                        method_name,
                        f"data_{kwargs['global']['trajectory_part']}_{trajectory_id}.pkl",
                    ),
                    "rb",
                ) as f:
                    goal_distribution = pickle.load(f)["goal_distribution"]
            else:
                goal_distribution = self.goal_module.find_goals(
                    dataset.scene_graph,
                    dataset.room_labels,
                    past_trajectory_info["sequence"],
                    past_trajectory,
                )

            # PSP module
            coarse_paths = self.planning_module.find_paths(
                dataset.scene_graph, past_trajectory, goal_distribution
            )

            if (
                "use_precomputed_kdes" in kwargs["global"]
                and kwargs["global"]["use_precomputed_kdes"]
                and os.path.exists(
                    os.path.join(
                        kwargs["output"]["folder"],
                        kwargs["global"]["scene"],
                        "out_data",
                        method_name,
                        f"data_{kwargs['global']['trajectory_part']}_{trajectory_id}.pkl",
                    )
                )
            ):
                logging.info("Using precomputed interaction tree.")
                with open(
                    os.path.join(
                        kwargs["output"]["folder"],
                        kwargs["global"]["scene"],
                        "out_data",
                        method_name,
                        f"data_{kwargs['global']['trajectory_part']}_{trajectory_id}.pkl",
                    ),
                    "rb",
                ) as f:
                    precomp_kdes = pickle.load(f)["kdes"]
            else:
                precomp_kdes = None

            compute_steady_state = (
                kwargs["global"]["compute_steady_state"]
                if "compute_steady_state" in kwargs["global"]
                else False
            )

            smooth_paths, predicted_time_horizon = self.low_level_module.process(
                dataset.scene_graph,
                past_trajectory,
                coarse_paths,
                time_horizon,
                goal_distribution=goal_distribution,
                eval_timesteps=eval_timesteps,
                precomp_kdes=precomp_kdes,
                compute_steady_state=compute_steady_state,
            )

            if smooth_paths:
                trajectory_probabilities = [i.probability for i in smooth_paths]
            else:
                trajectory_probabilities = None

            if self.visualization:
                fig = dsg.plot_scene_graph(
                    dataset.scene_graph, layer_settings=LAYER_SETTINGS
                )
                fig = visualize_paths_in_dsg(
                    fig,
                    [p.path for p in smooth_paths],
                    past_trajectory,
                    gt_future_trajectory,
                )
                fig.show()

            if (
                "deterministic" in kwargs["global"]
                and kwargs["global"]["deterministic"]
            ):
                smooth_paths.kdes = None

            out[trajectory_id] = {
                "eval_timesteps": eval_timesteps,  # np.ndarray
                "past_trajectory": past_trajectory,  # pd.DataFrame
                "past_trajectory_info": past_trajectory_info,  # dict
                "gt_future_trajectory": gt_future_trajectory,  # pd.DataFrame
                "future_trajectory_info": future_trajectory_info,  # dict
                "goal_distribution": goal_distribution,  # GoalSequences
                "pred_graph_paths": coarse_paths,  # CoarsePaths
                "trajectory_probabilities": trajectory_probabilities,  # Optional[list[float]]
                "pred_trajectories_interpolated": smooth_paths.paths,  # list[SmoothPath]
                "steady_state": smooth_paths.steady_state,  # Optional[np.ndarray]
                "kdes": smooth_paths.kdes,  # Optional[list[dict]]    (of format {"t": eval_timestep, "samples": kde_samples, "sample_weights": kde_sample_weights})
                "predicted_time_horizon": predicted_time_horizon,  # float
            }

            if not os.path.exists(
                os.path.join(
                    kwargs["output"]["folder"],
                    kwargs["global"]["scene"],
                    "out_data",
                    kwargs["global"]["method"],
                )
            ):
                os.makedirs(
                    os.path.join(
                        kwargs["output"]["folder"],
                        kwargs["global"]["scene"],
                        "out_data",
                        kwargs["global"]["method"],
                    )
                )
            with open(
                os.path.join(
                    kwargs["output"]["folder"],
                    kwargs["global"]["scene"],
                    "out_data",
                    kwargs["global"]["method"],
                    f"data_{kwargs['global']['trajectory_part']}_{trajectory_id}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(out[trajectory_id], f)

            print("Processed trajectory: ", trajectory_id)

        return out
