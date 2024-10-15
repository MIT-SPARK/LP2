import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import logging
from dataclasses import dataclass, field
from typing import Optional, Union
import networkx as nx

from sklearn.neighbors import KernelDensity
from lhmp.utils.method import NodeTypesTransitionTree


def reindex_trajectory(
    df: pd.DataFrame,
    new_timesteps: np.ndarray,
    linear_keys: list[str] = ["x", "y"],
    nearest_keys: list[str] = ["room_id"],
):
    data = {"t": new_timesteps}

    for key in linear_keys:
        try:
            data[key] = interp1d(
                df["t"],
                df[key],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )(new_timesteps)
        except Exception as e:
            print(e)
        if np.isnan(data[key][0]):
            data[key][0] = df[key].iloc[0]
        if np.isnan(data[key][-1]):
            data[key][-1] = df[key].iloc[-1]
        if sum(np.isnan(data[key])) > 0:
            print(key, "has nan values")
            logging.warning(f"key {key} has nan values")

    for key in nearest_keys:
        data[key] = interp1d(
            df["t"],
            df[key],
            kind="nearest",
            bounds_error=False,
            fill_value="extrapolate",
        )(new_timesteps)

        if np.isnan(data[key][0]):
            data[key][0] = df[key].iloc[0]
        if np.isnan(data[key][-1]):
            data[key][-1] = df[key].iloc[-1]

        if data[key].dtype == "float64":
            data[key] = data[key].astype(int)

    new_df = pd.DataFrame(data)

    return new_df


#### dataclasses: ####


@dataclass
class Goal:
    id: int
    semantic_label: str
    room_id: int
    probability: float
    duration: float


@dataclass
class GoalSequence:
    goals: list[Goal] = field(default_factory=list, init=True)
    probability: float = field(init=False)

    def __iter__(self):
        return iter(self.goals)

    def __post_init__(self):
        self.probability = np.prod([goal.probability for goal in self.goals])


@dataclass
class Tree:
    data: nx.DiGraph = field(default_factory=nx.DiGraph)
    root: int = field(default=0)


@dataclass
class GoalSequences:
    sequences: list[GoalSequence] = field(default_factory=list)
    tree: Optional[Tree] = None

    def __post_init__(self):
        if self.tree is not None and not self.sequences:
            self.sequences = self._get_sequences_from_tree()

    def __iter__(self):
        return iter(self.sequences)

    def _get_sequences_from_tree(self):
        sequences = []
        leaves = []
        for node in self.tree.data.nodes:
            if self.tree.data.out_degree(node) == 0:
                leaves.append(node)
        for leaf in leaves:
            sequence = []
            parent = leaf
            while parent != self.tree.root:
                sequence.append(self._goal_from_node(parent))
                parent = next(self.tree.data.predecessors(parent))
            sequence.append(self._goal_from_node(self.tree.root))
            sequence = sequence[::-1]
            sequences.append(GoalSequence(goals=sequence))

        return sequences

    def _goal_from_node(self, node: int):
        node_info = self.tree.data.nodes[node]
        id = node_info["id"]
        semantic_label = node_info["semantic_label"]
        room_id = node_info["room_id"]
        probability = node_info["probability"]
        duration = node_info["duration"]
        return Goal(id, semantic_label, room_id, probability, duration)


@dataclass
class CoarsePath:
    path: list[list[int]] = field(default_factory=list)
    segment_probabilities: list[float] = field(default_factory=list)
    probability: float = field(init=False)

    def __post_init__(self):
        self.probability = np.prod(self.segment_probabilities)

    def __iter__(self):
        return iter(self.path)


@dataclass
class CoarsePaths:
    paths: list[CoarsePath] = field(default_factory=list)
    transition_tree: Optional[Tree] = None

    def __post_init__(self):
        if self.transition_tree is not None and not self.paths:
            self.paths = self._get_paths_from_tree()

    def __iter__(self):
        return iter(self.paths)

    def _get_paths_from_tree(self):
        paths = []
        leaves = []
        for node in self.transition_tree.data.nodes:
            if self.transition_tree.data.out_degree(node) == 0:
                leaves.append(node)
        for leaf in leaves:
            path = []
            segment_probabilities = []
            parent = leaf
            parent = next(self.transition_tree.data.predecessors(parent))
            path_segment = []
            while parent != self.transition_tree.root:
                if (
                    self.transition_tree.data.nodes[parent]["type"]
                    == NodeTypesTransitionTree.INCOMING
                ):
                    path_segment = []
                if (
                    self.transition_tree.data.nodes[parent]["type"]
                    == NodeTypesTransitionTree.PASSING
                ):
                    path_segment.append(self._goal_from_node(parent))
                    if (
                        self.transition_tree.data.nodes[
                            next(self.transition_tree.data.predecessors(parent))
                        ]["type"]
                        == NodeTypesTransitionTree.OUTGOING
                    ):
                        interaction_probability_edge = self.transition_tree.data.edges[
                            (
                                next(self.transition_tree.data.predecessors(parent)),
                                parent,
                            )
                        ]
                        assert (
                            interaction_probability_edge["edge_type"]
                            == "interaction_probability"
                        )
                        probability = interaction_probability_edge["transition_rate"]
                        segment_probabilities.append(probability)
                if (
                    self.transition_tree.data.nodes[parent]["type"]
                    == NodeTypesTransitionTree.PROBABILITY
                ):
                    interaction_probability_edge = self.transition_tree.data.edges[
                        (next(self.transition_tree.data.predecessors(parent)), parent)
                    ]
                    assert (
                        interaction_probability_edge["edge_type"]
                        == "interaction_probability"
                    )
                    probability = interaction_probability_edge["transition_rate"]
                    segment_probabilities.append(probability)
                if (
                    self.transition_tree.data.nodes[parent]["type"]
                    == NodeTypesTransitionTree.OUTGOING
                ):
                    if path_segment:
                        path_segment = path_segment[::-1]
                        path.append(path_segment)
                    else:
                        print("empty path segment")
                parent = next(self.transition_tree.data.predecessors(parent))
            path = path[::-1]
            paths.append(
                CoarsePath(path=path, segment_probabilities=segment_probabilities)
            )

        return sorted(paths, key=lambda x: x.probability, reverse=True)

    def _goal_from_node(self, node: int):
        node_info = self.transition_tree.data.nodes[node]
        return node_info["places_node_id"]


@dataclass
class SmoothPath:
    path: pd.DataFrame
    probability: float


@dataclass
class SmoothPaths:
    paths: list[SmoothPath] = field(default_factory=list)
    predicted_time_horizon: float = None
    kdes: Optional[list[dict]] = None
    steady_state: Optional[np.ndarray] = None

    def __iter__(self):
        return iter(self.paths)
