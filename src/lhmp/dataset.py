import os
import json
from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union

import pandas as pd

import spark_dsg as dsg
from spark_dsg._dsg_bindings import DynamicSceneGraph, NodeSymbol

from lhmp.utils.general import read_json


def load_dataset(
    scene_graph_path: str,
    room_labels_path: str,
    trajectory_dir: str,
    time_horizon: float,
    alpha: float = None,
    beta: float = None,
    n_past_interactions: int = None,
    n_total_interactions: int = None,
    limit_time_horizon: bool = False,
    time_after_interaction: float = None,
    future_trajectory_length: float = None,
    *args,
    **kwargs,
):

    assert (
        (alpha and beta and time_horizon)
        or (n_past_interactions and n_total_interactions)
        or (n_past_interactions and time_after_interaction and future_trajectory_length)
    ), "Either alpha and beta or n_past_interactions must be provided"
    if alpha and beta and time_horizon:
        assert alpha + beta <= 1.0, "Alpha and beta must sum to less than 1.0"
    if n_past_interactions and n_total_interactions:
        assert (
            n_past_interactions < n_total_interactions
        ), "Number of past interactions must be less than total number of interactions"

    dataset = LHMPDataset(
        scene_graph_path,
        room_labels_path,
        trajectory_dir,
        time_horizon=time_horizon,
        alpha=alpha,
        beta=beta,
        limit_time_horizon=limit_time_horizon,
        n_past_interactions=n_past_interactions,
        n_total_interactions=n_total_interactions,
        time_after_interaction=time_after_interaction,
        future_trajectory_length=future_trajectory_length,
    )

    return dataset


class LHMPDataset:
    """
    Loads data for the LHMP pipeline
    :param scene_graph_path: path to scene graph JSON file
    :param room_labels_path: path to room labels JSON file
    :param trajectory_dir: path to directory containing trajectory CSV files
    :param n_past_interactions: number of interactions to include in past trajectory
    """

    scene_graph: DynamicSceneGraph
    room_labels: dict
    full_trajectories: list[pd.DataFrame]
    past_trajectories: list[pd.DataFrame]
    past_trajectories_info: list[dict]
    future_trajectories: list[pd.DataFrame]
    future_trajectories_info: list[dict]

    def __init__(
        self,
        scene_graph_path: str,
        room_labels_path: str,
        trajectory_dir: str,
        time_horizon: float = None,
        alpha: float = None,
        beta: float = None,
        limit_time_horizon: bool = False,
        n_past_interactions: int = None,
        n_total_interactions: int = None,
        time_after_interaction: float = None,
        future_trajectory_length: float = None,
    ) -> None:
        self.scene_graph = self.load_scene_graph(scene_graph_path)
        self.room_labels = self.load_room_labels(room_labels_path)
        self.full_trajectories = [
            self.load_trajectory(os.path.join(trajectory_dir, file))
            for file in sorted(
                os.listdir(trajectory_dir),
                key=lambda x: int(x.split(".")[0].split("_")[-1]),
            )
        ]
        if n_past_interactions and time_after_interaction and future_trajectory_length:
            (
                self.past_trajectories,
                self.past_trajectories_info,
                self.future_trajectories,
                self.future_trajectories_info,
            ) = zip(
                *[
                    self.split_trajectory_after_interaction(
                        trajectory,
                        n_past_interactions,
                        time_after_interaction,
                        future_trajectory_length,
                        limit_time_horizon,
                    )
                    for trajectory in self.full_trajectories
                ]
            )
        elif n_past_interactions and n_total_interactions:
            (
                self.past_trajectories,
                self.past_trajectories_info,
                self.future_trajectories,
                self.future_trajectories_info,
            ) = zip(
                *[
                    self.split_trajectory(
                        trajectory, n_past_interactions, n_total_interactions
                    )
                    for trajectory in self.full_trajectories
                ]
            )
        elif time_horizon and alpha and beta:
            (
                self.past_trajectories,
                self.past_trajectories_info,
                self.future_trajectories,
                self.future_trajectories_info,
            ) = zip(
                *[
                    self.split_trajectory_time(
                        trajectory, time_horizon, alpha, beta, limit_time_horizon
                    )
                    for trajectory in self.full_trajectories
                ]
            )

    def __len__(self) -> int:
        return len(self.full_trajectories)

    def __getitem__(self, idx: int) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
        return (
            idx,
            self.past_trajectories[idx],
            self.past_trajectories_info[idx],
            self.future_trajectories[idx],
            self.future_trajectories_info[idx],
        )

    def load_scene_graph(self, scene_graph_path: str):
        """
        Load scene graph from JSON file
        """
        assert os.path.exists(scene_graph_path), "Scene graph path does not exist"
        scene_graph = dsg.DynamicSceneGraph.load(scene_graph_path)
        return scene_graph

    def load_room_labels(self, room_labels_path: str):
        """
        Load room labels from JSON file
        """
        assert os.path.exists(room_labels_path), "Room labels path does not exist"
        room_labels = read_json(room_labels_path)
        return room_labels

    def load_trajectory(self, trajectory_path: str):
        """
        Load trajectory from CSV file
        """
        assert os.path.exists(trajectory_path), "Trajectory path does not exist"
        trajectory = pd.read_csv(trajectory_path, index_col=0)
        return trajectory

    def split_trajectory_time(
        self,
        trajectory: pd.DataFrame,
        time_horizon: float,
        alpha: float,
        beta: float,
        limit_time_horizon: bool = False,
    ) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
        """
        split trajectory into past and future based on time horizon, alpha and beta
        :param trajectory: full trajectory
        :param time_horizon: time horizon of full trajectory
        :param alpha: length of past trajectory as fraction of time horizon
        :param beta: length of future trajectory as fraction of time horizon
        """

        past_trajectory = trajectory[trajectory["t"] <= time_horizon * alpha]
        future_trajectory = trajectory[trajectory["t"] > time_horizon * alpha]

        if limit_time_horizon:
            future_trajectory = future_trajectory[
                future_trajectory["t"] <= time_horizon * (alpha + beta)
            ]

        past_activities = get_activities(past_trajectory)
        future_activities = get_activities(future_trajectory)

        past_trajectory_info = self.get_trajecory_info(past_trajectory, past_activities)
        future_trajectory_info = self.get_trajecory_info(
            future_trajectory, future_activities
        )

        return (
            past_trajectory,
            past_trajectory_info,
            future_trajectory,
            future_trajectory_info,
        )

    def split_trajectory(
        self,
        trajectory: pd.DataFrame,
        n_past_interactions: int,
        n_total_interactions: int,
    ) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
        """
        Extract past trajectory from full trajectory
        :param trajectory: full trajectory
        :param time_horizon: time horizon of full trajectory
        :param n_past_interactions: number of interactions to include in past trajectory
        :param n_total_interactions: total number of interactions to include in past and future trajectory
        """

        activities = get_activities(trajectory)

        idx = 0
        interaction_idx = 0
        past_trajectory = None
        future_trajectory = None
        for activity_idx, activity in enumerate(activities):
            if interaction_idx == n_past_interactions and past_trajectory is None:
                past_trajectory = trajectory.iloc[: idx + 1]
                past_activities = activities[:activity_idx]
            if interaction_idx >= n_total_interactions:
                future_trajectory = trajectory.iloc[
                    past_trajectory.iloc[-1].name + 1 : idx + 1
                ]
                future_activities = activities[len(past_activities) : activity_idx]
                break  # TODO: verify that this does not cause issues.
            if activity == -1:
                while trajectory["interaction_id"].iloc[idx] == -1:
                    idx += 1
                    if idx == len(trajectory):
                        break
            else:
                while activity == trajectory["interaction_id"].iloc[idx]:
                    idx += 1
                    if idx == len(trajectory):
                        break
                if idx == len(trajectory):
                    future_trajectory = trajectory.iloc[
                        past_trajectory.iloc[-1].name + 1 :
                    ]
                    future_activities = activities[len(past_activities) :]
                    break
                interaction_idx += 1

        assert (
            past_trajectory is not None and future_trajectory is not None
        ), "Not enough interactions in trajectory, failed to split data"

        past_trajectory_info = self.get_trajecory_info(past_trajectory, past_activities)
        future_trajectory_info = self.get_trajecory_info(
            future_trajectory, future_activities
        )

        return (
            past_trajectory,
            past_trajectory_info,
            future_trajectory,
            future_trajectory_info,
        )

    def split_trajectory_after_interaction(
        self,
        trajectory: pd.DataFrame,
        n_past_interactions: int,
        time_after_interaction: float,
        future_trajectory_length: float,
        limit_time_horizon: bool = False,
    ) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
        """
        Split trajectory such that
        * future trajectory is at least *future_trajectory_length* long
        * past trajectory contains *n_past_interactions* interactions
        * split *time_after_interaction* seconds after the last past interaction
        """
        activities = get_activities(trajectory)

        idx = 0
        interaction_idx = 0
        past_trajectory = None
        future_trajectory = None
        for activity_idx, activity in enumerate(activities):
            if interaction_idx == n_past_interactions and past_trajectory is None:
                time = trajectory.iloc[idx]["t"]
                cut_off_time = time + time_after_interaction
                if trajectory.iloc[-1]["t"] <= cut_off_time + future_trajectory_length:
                    past_trajectory = trajectory[
                        trajectory["t"]
                        <= trajectory.iloc[-1]["t"] - future_trajectory_length
                    ]
                    past_activities = get_activities(past_trajectory)
                    future_trajectory = trajectory[
                        trajectory["t"]
                        > trajectory.iloc[-1]["t"] - future_trajectory_length
                    ]
                    future_activities = get_activities(future_trajectory)
                else:
                    past_trajectory = trajectory[trajectory["t"] <= cut_off_time]
                    past_activities = get_activities(past_trajectory)
                    future_trajectory = trajectory[trajectory["t"] > cut_off_time]
                    future_activities = get_activities(future_trajectory)
                break
            while activity == trajectory["interaction_id"].iloc[idx]:
                idx += 1
                if idx == len(trajectory):
                    print(
                        "dataset -- split_trajectory_after_interaction: reached end of trajectory"
                    )
                    break
            if activity != -1:
                interaction_idx += 1

        if past_trajectory is None or future_trajectory is None:
            past_trajectory = trajectory[
                trajectory["t"] <= trajectory.iloc[-1]["t"] - future_trajectory_length
            ]
            past_activities = get_activities(past_trajectory)
            future_trajectory = trajectory[
                trajectory["t"] > trajectory.iloc[-1]["t"] - future_trajectory_length
            ]
            future_activities = get_activities(future_trajectory)
        past_trajectory_info = self.get_trajecory_info(past_trajectory, past_activities)
        future_trajectory_info = self.get_trajecory_info(
            future_trajectory, future_activities
        )

        return (
            past_trajectory,
            past_trajectory_info,
            future_trajectory,
            future_trajectory_info,
        )

    def get_trajecory_info(
        self, trajectory: pd.DataFrame, activities: list[int]
    ) -> dict:
        if "object" in trajectory.columns:
            return self.get_trajecory_info_with_actions(trajectory, activities)
        else:
            return self.get_trajecory_info_without_actions(trajectory, activities)

    def get_trajecory_info_without_actions(
        self, trajectory: pd.DataFrame, activities: list[int]
    ) -> dict:
        node_info = lambda node, room_id, duration: {
            "id": node.id.category_id,
            "semantic_label": node.attributes.name,
            "layer": node.layer,
            "room_id": room_id,
            "duration": duration,
        }
        room_ids = []
        durations = []
        idx = 0
        for activity in activities:
            start = idx
            while activity == trajectory["interaction_id"].iloc[idx]:
                idx += 1
                if idx == len(trajectory):
                    break
            end = idx - 1
            if trajectory["interaction_id"].iloc[start] == -1:
                continue
            room_ids.append(trajectory["room_id"].iloc[start])
            duration = trajectory["t"].iloc[end] - trajectory["t"].iloc[start]
            durations.append(duration)
            if idx == len(trajectory):
                break
        trajectory_info = {
            "sequence": [
                node_info(
                    self.scene_graph.get_node(NodeSymbol("O", cat).value),
                    room_id,
                    duration,
                )
                for cat, room_id, duration in zip(
                    [i for i in activities if i != -1], room_ids, durations
                )
            ]
        }

        return trajectory_info

    def get_trajecory_info_with_actions(
        self, trajectory: pd.DataFrame, activities: list[int]
    ) -> dict:

        node_info = lambda node, action, room_id, duration: {
            "id": node.id.category_id,
            "semantic_label": node.attributes.name,
            "action": action,
            "layer": node.layer,
            "room_id": room_id,
            "duration": duration,
        }
        room_ids = []
        durations = []
        actions = []

        idx = 0
        for activity in activities:
            start = idx
            while activity == trajectory["interaction_id"].iloc[idx]:
                idx += 1
                if idx == len(trajectory):
                    break
            end = idx - 1
            if trajectory["interaction_id"].iloc[start] == -1:
                continue
            room_ids.append(trajectory["room_id"].iloc[start])
            duration = trajectory["t"].iloc[end] - trajectory["t"].iloc[start]
            durations.append(duration)
            act = trajectory["action"].iloc[start]
            actions.append(act)
            if idx == len(trajectory):
                break
        trajectory_info = {
            "sequence": [
                node_info(
                    self.scene_graph.get_node(NodeSymbol("O", cat).value),
                    action,
                    room_id,
                    duration,
                )
                for cat, action, room_id, duration in zip(
                    [i for i in activities if i != -1], actions, room_ids, durations
                )
            ]
        }

        return trajectory_info


def get_activities(trajectory: pd.DataFrame) -> list[int]:
    activities = [
        (element)
        for (idx, element) in enumerate(trajectory["interaction_id"])
        if (trajectory["interaction_id"].iloc[idx - 1] != element or idx == 0)
    ]
    return activities
