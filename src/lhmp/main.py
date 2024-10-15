from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union
import pandas as pd
import json
import numpy as np
import yaml
import pickle
import os
import argparse
import logging
from mergedeep import merge

import spark_dsg as dsg
import plotly.graph_objects as go

from importlib import import_module

from lhmp.evaluation import Evaluation
from lhmp.pipeline import Pipeline
from lhmp.utils.visualization import (
    visualize_path,
    visualize_paths_in_dsg,
    visualize_dsg_layer,
    plot_trajectory,
    animation_vis,
    animation_vis_unstructured,
    animation_vis_discrete,
    LAYER_SETTINGS,
)
from lhmp.utils.dsg import Layers
from lhmp.utils.general import load_module


def main(
    global_config_file="cfg/global_config.yaml",
    method_config_file="cfg/method_configs/project_config_LP2.yaml",
    scene_config_file="cfg/scene_configs/scene_config_office.yaml",
    run_pipeline: bool = True,
):
    with open(global_config_file, "r") as f:
        global_cfg = yaml.safe_load(f)

    with open(method_config_file, "r") as f:
        method_cfg = yaml.safe_load(f)

    with open(scene_config_file, "r") as f:
        scene_cfg = yaml.safe_load(f)

    cfg = merge(global_cfg, scene_cfg, method_cfg)

    GoalModule = load_module(cfg["goal_module"]["class"])
    PlanningModule = load_module(cfg["planning_module"]["class"])
    LowLevelModule = load_module(cfg["low_level_module"]["class"])

    part = cfg["global"]["trajectory_part"]
    if run_pipeline:
        if (
            "interaction_time_ablation" in cfg["global"]
            and cfg["global"]["interaction_time_ablation"]
        ):
            raise NotImplementedError(
                "Interaction time ablation not meant for running pipeline. Either set run_pipeline to False or choose different config."
            )

        logging.info(
            f"\n############################################\nRunning Pipeline, method: {cfg['global']['method']}\n############################################\n"
        )
        pipeline = Pipeline(GoalModule, PlanningModule, LowLevelModule, **cfg)

        if (
            "gt_action_sequences_ablation" in cfg["global"]
            and cfg["global"]["gt_action_sequences_ablation"]
        ):
            PipelineClass = load_module(cfg["pipeline"]["class"])
            pipeline = PipelineClass(GoalModule, PlanningModule, LowLevelModule, **cfg)

        data_out = pipeline.run(**cfg, **cfg["data"], **cfg["global"])

        # save output of run to file
        with open(
            os.path.join(
                cfg["output"]["folder"],
                cfg["global"]["scene"],
                "out_data",
                cfg["global"]["method"],
                f"p{part}_data.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(data_out, f)

        print("predictions saved.")
    else:
        logging.info(
            f"Skipping Pipeline, loading output of last run, method: {cfg['global']['method']}\n"
        )

        if (
            "interaction_time_ablation" in cfg["global"]
            and cfg["global"]["interaction_time_ablation"]
        ):
            PipelineClass = load_module(cfg["pipeline"]["class"])
            pipeline = PipelineClass(GoalModule, PlanningModule, LowLevelModule, **cfg)

            with open(
                os.path.join(
                    cfg["output"]["folder"],
                    cfg["global"]["scene"],
                    "out_data",
                    cfg["global"]["underlying_method"],
                    f"p{part}_data.pkl",
                ),
                "rb",
            ) as f:
                data_out = pickle.load(f)
            data_out = pipeline.run(data_out, **cfg, **cfg["data"], **cfg["global"])
        else:
            with open(
                os.path.join(
                    cfg["output"]["folder"],
                    cfg["global"]["scene"],
                    "out_data",
                    cfg["global"]["method"],
                    f"p{part}_data.pkl",
                ),
                "rb",
            ) as f:
                data_out = pickle.load(f)

    eval = Evaluation(**cfg["global"], **cfg["evaluation"])

    for trajectory_id, trajectory_data in data_out.items():
        # visualize places layer with paths:
        print(f"### evaluating trajectory {trajectory_id} ###")
        scene_graph_path = cfg["data"]["scene_graph_path"]
        scene_graph = dsg.DynamicSceneGraph.load(scene_graph_path)
        places_layer = scene_graph.get_layer(Layers.PLACES)

        vis_bool = "visualization" in cfg["global"] and cfg["global"]["visualization"]

        if vis_bool:
            fig = go.Figure()
            fig = visualize_dsg_layer(fig, places_layer, marker_size=3)
            fig = visualize_paths_in_dsg(
                fig,
                trajectory_data["pred_trajectories_interpolated"],
                trajectory_data["past_trajectory"],
                trajectory_data["gt_future_trajectory"],
                offset=0,
            )
            fig.show()

        if trajectory_data["pred_graph_paths"] and vis_bool:
            fig = go.Figure()
            fig = visualize_dsg_layer(fig, places_layer, marker_size=3)
            fig = plot_trajectory(
                fig,
                trajectory_data["past_trajectory"],
                color="rgb(255,190,0)",
                offset=0,
            )
            fig = plot_trajectory(
                fig, trajectory_data["gt_future_trajectory"], offset=0
            )
            for path in trajectory_data["pred_graph_paths"]:
                for segment in path:
                    fig = visualize_path(fig, places_layer, segment, offset=False)
            fig.show()

        if cfg["global"]["animate"]:
            if (
                "discrete_animation" not in cfg["global"]
                or not cfg["global"]["discrete_animation"]
            ):
                if trajectory_data["kdes"]:
                    if scene_cfg["animation"]["visualize_bg"]:
                        with open(scene_cfg["animation"]["bg_cfg"], "r") as f:
                            bg_cfg = yaml.safe_load(f)
                        animation = animation_vis_unstructured(
                            places_layer,
                            trajectory_data["kdes"],
                            trajectory_data["pred_trajectories_interpolated"],
                            trajectory_data["gt_future_trajectory"],
                            cfg["global"]["environment_bounds"],
                            1.0,
                            90.0,
                            bg_cfg,
                            bandwidth=cfg["global"]["bandwidth"],
                        )
                    else:
                        animation = animation_vis_unstructured(
                            places_layer,
                            trajectory_data["kdes"],
                            trajectory_data["pred_trajectories_interpolated"],
                            trajectory_data["gt_future_trajectory"],
                            cfg["global"]["environment_bounds"],
                            1.0,
                            90.0,
                            bandwidth=cfg["global"]["bandwidth"],
                        )
                else:
                    if scene_cfg["animation"]["visualize_bg"]:
                        with open(scene_cfg["animation"]["bg_cfg"], "r") as f:
                            bg_cfg = yaml.safe_load(f)
                        animation = animation_vis(
                            places_layer,
                            trajectory_data["pred_trajectories_interpolated"],
                            trajectory_data["gt_future_trajectory"],
                            cfg["global"]["environment_bounds"],
                            0.1,
                            60.0,
                            bg_cfg,
                        )
                    else:
                        animation = animation_vis(
                            places_layer,
                            trajectory_data["pred_trajectories_interpolated"],
                            trajectory_data["gt_future_trajectory"],
                            cfg["global"]["environment_bounds"],
                            0.1,
                            60.0,
                            None,
                        )

                if not os.path.exists(
                    os.path.join(
                        cfg["output"]["folder"],
                        cfg["global"]["scene"],
                        "animation",
                        cfg["global"]["method"],
                    )
                ):
                    os.makedirs(
                        os.path.join(
                            cfg["output"]["folder"],
                            cfg["global"]["scene"],
                            "animation",
                            cfg["global"]["method"],
                        )
                    )
                animation.save(
                    os.path.join(
                        cfg["output"]["folder"],
                        cfg["global"]["scene"],
                        "animation",
                        cfg["global"]["method"],
                        f"kde_animation_{cfg['global']['method']}_{part}_{trajectory_id}.gif",
                    ),
                    writer="pillow",
                    fps=5,
                )
            else:
                if (
                    trajectory_data["pred_trajectories_interpolated"]
                    and scene_cfg["animation"]["visualize_bg"]
                ):
                    with open(scene_cfg["animation"]["bg_cfg"], "r") as f:
                        bg_cfg = yaml.safe_load(f)
                    animation_discrete = animation_vis_discrete(
                        trajectory_data["pred_trajectories_interpolated"],
                        trajectory_data["past_trajectory"],
                        trajectory_data["gt_future_trajectory"],
                        0.0,
                        60.0,
                        bg_cfg,
                    )
                    if not os.path.exists(
                        os.path.join(
                            cfg["output"]["folder"],
                            cfg["global"]["scene"],
                            "animation_paths",
                            cfg["global"]["method"],
                        )
                    ):
                        os.makedirs(
                            os.path.join(
                                cfg["output"]["folder"],
                                cfg["global"]["scene"],
                                "animation_paths",
                                cfg["global"]["method"],
                            )
                        )
                    animation_discrete.save(
                        os.path.join(
                            cfg["output"]["folder"],
                            cfg["global"]["scene"],
                            "animation_paths",
                            cfg["global"]["method"],
                            f"path_animation_{cfg['global']['method']}_{part}_{trajectory_id}.gif",
                        ),
                        writer="pillow",
                        fps=5,
                    )

        if "save_trajectories" in cfg["global"] and cfg["global"]["save_trajectories"]:
            if not os.path.exists(
                f"visualization/trajectories_out/traj_{part}_{trajectory_id}"
            ):
                os.mkdir(f"visualization/trajectories_out/traj_{part}_{trajectory_id}")
            trajectory_data["past_trajectory"][["x", "y", "z", "t"]].to_csv(
                f"visualization/trajectories_out/traj_{part}_{trajectory_id}/past_trajectory.csv"
            )
            trajectory_data["gt_future_trajectory"][["x", "y", "z", "t"]].to_csv(
                f"visualization/trajectories_out/traj_{part}_{trajectory_id}/gt_future_trajectory.csv"
            )
            for i, pred_trajectory in enumerate(
                trajectory_data["pred_trajectories_interpolated"]
            ):
                z_ = (
                    np.ones_like(np.array(pred_trajectory["x"]))
                    * trajectory_data["past_trajectory"]["z"].iloc[-1]
                )
                pred_trajectory["z"] = z_
                pred_trajectory = pred_trajectory[["x", "y", "z", "t"]]
                pred_trajectory.to_csv(
                    f"visualization/trajectories_out/traj_{part}_{trajectory_id}/pred_trajectory_{i}.csv"
                )

        for idx, N_BoN in enumerate(cfg["evaluation"]["N_BoN"]):

            if idx == 0:
                (
                    res_nll,
                    res_ade_fde_at_t,
                    interaction_prediction_likelihood,
                    semantic_interaction_prediction_likelihood,
                    prediction_top_n_accuracy_by_id,
                    prediction_top_n_accuracy_by_semantic_class,
                    room_prediction_accs,
                    steady_state_distance_js,
                    steady_state_distance_tv,
                ) = eval.evaluate_trajectory(
                    trajectory_data,
                    evaluate_interaction_prediction="evaluate_interaction_prediction"
                    in cfg["global"]
                    and cfg["global"]["evaluate_interaction_prediction"],
                    evaluate_time_horizon_steady_state=True,
                    N_BoN=N_BoN,
                )

                if not os.path.exists(
                    os.path.join(
                        cfg["output"]["folder"],
                        cfg["global"]["scene"],
                        "nll_t",
                        cfg["global"]["method"],
                    )
                ):
                    os.makedirs(
                        os.path.join(
                            cfg["output"]["folder"],
                            cfg["global"]["scene"],
                            "nll_t",
                            cfg["global"]["method"],
                        )
                    )
                if res_nll is not None:
                    res_nll.to_csv(
                        os.path.join(
                            cfg["output"]["folder"],
                            cfg["global"]["scene"],
                            "nll_t",
                            cfg["global"]["method"],
                            f"nll_{part}_{trajectory_id}.csv",
                        )
                    )

                if interaction_prediction_likelihood is not None:
                    if not os.path.exists(
                        os.path.join(
                            cfg["output"]["folder"],
                            cfg["global"]["scene"],
                            "interaction_prediction_likelihood",
                            cfg["global"]["method"],
                        )
                    ):
                        os.makedirs(
                            os.path.join(
                                cfg["output"]["folder"],
                                cfg["global"]["scene"],
                                "interaction_prediction_likelihood",
                                cfg["global"]["method"],
                            )
                        )
                    interaction_prediction_likelihood.to_csv(
                        os.path.join(
                            cfg["output"]["folder"],
                            cfg["global"]["scene"],
                            "interaction_prediction_likelihood",
                            cfg["global"]["method"],
                            f"interaction_prediction_likelihood_{part}_{trajectory_id}.csv",
                        )
                    )

                if semantic_interaction_prediction_likelihood is not None:
                    if not os.path.exists(
                        os.path.join(
                            cfg["output"]["folder"],
                            cfg["global"]["scene"],
                            "semantic_interaction_prediction_likelihood",
                            cfg["global"]["method"],
                        )
                    ):
                        os.makedirs(
                            os.path.join(
                                cfg["output"]["folder"],
                                cfg["global"]["scene"],
                                "semantic_interaction_prediction_likelihood",
                                cfg["global"]["method"],
                            )
                        )
                    semantic_interaction_prediction_likelihood.to_csv(
                        os.path.join(
                            cfg["output"]["folder"],
                            cfg["global"]["scene"],
                            "semantic_interaction_prediction_likelihood",
                            cfg["global"]["method"],
                            f"semantic_interaction_prediction_likelihood_{part}_{trajectory_id}.csv",
                        )
                    )

                if prediction_top_n_accuracy_by_id is not None:
                    if not os.path.exists(
                        os.path.join(
                            cfg["output"]["folder"],
                            cfg["global"]["scene"],
                            "top_n_accuracy_by_id",
                            cfg["global"]["method"],
                        )
                    ):
                        os.makedirs(
                            os.path.join(
                                cfg["output"]["folder"],
                                cfg["global"]["scene"],
                                "top_n_accuracy_by_id",
                                cfg["global"]["method"],
                            )
                        )
                    with open(
                        os.path.join(
                            cfg["output"]["folder"],
                            cfg["global"]["scene"],
                            "top_n_accuracy_by_id",
                            cfg["global"]["method"],
                            f"top_{cfg['evaluation']['top_n_accuracy_N_id']}_accuracy_by_id_{part}_{trajectory_id}.json",
                        ),
                        "w",
                    ) as f:
                        json.dump(prediction_top_n_accuracy_by_id, f)

                if prediction_top_n_accuracy_by_semantic_class is not None:
                    if not os.path.exists(
                        os.path.join(
                            cfg["output"]["folder"],
                            cfg["global"]["scene"],
                            "top_n_accuracy_by_semantic_class",
                            cfg["global"]["method"],
                        )
                    ):
                        os.makedirs(
                            os.path.join(
                                cfg["output"]["folder"],
                                cfg["global"]["scene"],
                                "top_n_accuracy_by_semantic_class",
                                cfg["global"]["method"],
                            )
                        )
                    with open(
                        os.path.join(
                            cfg["output"]["folder"],
                            cfg["global"]["scene"],
                            "top_n_accuracy_by_semantic_class",
                            cfg["global"]["method"],
                            f"top_{cfg['evaluation']['top_n_accuracy_N_semantic']}_accuracy_by_semantic_class_{part}_{trajectory_id}.json",
                        ),
                        "w",
                    ) as f:
                        json.dump(prediction_top_n_accuracy_by_semantic_class, f)

                if steady_state_distance_js is not None:
                    if not os.path.exists(
                        os.path.join(
                            cfg["output"]["folder"],
                            cfg["global"]["scene"],
                            "steady_state_distance_js",
                            cfg["global"]["method"],
                        )
                    ):
                        os.makedirs(
                            os.path.join(
                                cfg["output"]["folder"],
                                cfg["global"]["scene"],
                                "steady_state_distance_js",
                                cfg["global"]["method"],
                            )
                        )
                    steady_state_distance_js.to_csv(
                        os.path.join(
                            cfg["output"]["folder"],
                            cfg["global"]["scene"],
                            "steady_state_distance_js",
                            cfg["global"]["method"],
                            f"steady_state_distance_js_{part}_{trajectory_id}.csv",
                        )
                    )

                if steady_state_distance_tv is not None:
                    if not os.path.exists(
                        os.path.join(
                            cfg["output"]["folder"],
                            cfg["global"]["scene"],
                            "steady_state_distance_tv",
                            cfg["global"]["method"],
                        )
                    ):
                        os.makedirs(
                            os.path.join(
                                cfg["output"]["folder"],
                                cfg["global"]["scene"],
                                "steady_state_distance_tv",
                                cfg["global"]["method"],
                            )
                        )
                    steady_state_distance_tv.to_csv(
                        os.path.join(
                            cfg["output"]["folder"],
                            cfg["global"]["scene"],
                            "steady_state_distance_tv",
                            cfg["global"]["method"],
                            f"steady_state_distance_tv_{part}_{trajectory_id}.csv",
                        )
                    )

            else:
                _, res_ade_fde_at_t, _, _, _, _, room_prediction_accs, _, _ = (
                    eval.evaluate_trajectory(
                        trajectory_data,
                        evaluate_interaction_prediction=False,
                        evaluate_nll=False,
                        evaluate_time_horizon_steady_state=False,
                        N_BoN=N_BoN,
                    )
                )

            if not os.path.exists(
                os.path.join(
                    cfg["output"]["folder"],
                    cfg["global"]["scene"],
                    f"ade_fde_Bo{N_BoN}",
                    cfg["global"]["method"],
                )
            ):
                os.makedirs(
                    os.path.join(
                        cfg["output"]["folder"],
                        cfg["global"]["scene"],
                        f"ade_fde_Bo{N_BoN}",
                        cfg["global"]["method"],
                    )
                )
            if res_ade_fde_at_t is not None:
                res_ade_fde_at_t.to_csv(
                    os.path.join(
                        cfg["output"]["folder"],
                        cfg["global"]["scene"],
                        f"ade_fde_Bo{N_BoN}",
                        cfg["global"]["method"],
                        f"ade_fde_{part}_{trajectory_id}.csv",
                    )
                )

            if not os.path.exists(
                os.path.join(
                    cfg["output"]["folder"],
                    cfg["global"]["scene"],
                    f"room_prediction_Bo{N_BoN}",
                    cfg["global"]["method"],
                )
            ):
                os.makedirs(
                    os.path.join(
                        cfg["output"]["folder"],
                        cfg["global"]["scene"],
                        f"room_prediction_Bo{N_BoN}",
                        cfg["global"]["method"],
                    )
                )
            if res_ade_fde_at_t is not None:
                room_prediction_accs.to_csv(
                    os.path.join(
                        cfg["output"]["folder"],
                        cfg["global"]["scene"],
                        f"room_prediction_Bo{N_BoN}",
                        cfg["global"]["method"],
                        f"room_prediction_{part}_{trajectory_id}.csv",
                    )
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_pipeline",
        type=bool,
        default=True,
        help="Whether to run the pipeline or not.",
    )
    parser.add_argument(
        "--global_config_file",
        type=str,
        default="cfg/global_config.yaml",
        help="Path to the global config file.",
    )
    parser.add_argument(
        "--method_config_file",
        type=str,
        default="cfg/method_configs/project_config_LP2.yaml",
        help="Path to the method config file.",
    )
    parser.add_argument(
        "--scene_config_file",
        type=str,
        default="cfg/scene_configs/scene_config_office.yaml",
        help="Path to the scene config file.",
    )
    args = parser.parse_args()

    main(
        args.global_config_file,
        args.method_config_file,
        args.scene_config_file,
        args.run_pipeline,
    )
