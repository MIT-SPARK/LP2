from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union
import pandas as pd

import networkx as nx
from scipy.spatial import cKDTree

from spark_dsg._dsg_bindings import NodeSymbol, DynamicSceneGraph as DSG

from lhmp.utils.visualization import LAYER_SETTINGS, visualize_path
from lhmp.utils.general import performance_measure, read_json, ensure_not_none
from lhmp.utils.dsg import Layers, get_closest_node, construct_places_layer_nx
from lhmp.utils.math import euclidean_distance
from lhmp.utils.method import NodeTypesTransitionTree
from lhmp.modules.planning.planning_module_base import PlanningModuleBase
from lhmp.dataset import LHMPDataset
from lhmp.utils.data import GoalSequences, CoarsePaths, CoarsePath, Tree


class CoarsePlacesTreePlanner(PlanningModuleBase):
    def __init__(self, *args, **kwargs):
        self.scene_graph = None
        self.places_layer_nx: Optional[nx.Graph] = None
        self.places_layer_nx_kdTree: Optional[cKDTree] = None

        self.DEFAULT_TIME = 0.1  # [s]

        if "avg_walking_speed" in kwargs:
            self.walking_speed = kwargs["avg_walking_speed"]
        else:
            self.walking_speed = 1.4

    def find_paths(
        self,
        scene_graph: DSG,
        past_trajectory: pd.DataFrame,
        goal_distribution: GoalSequences,
    ) -> CoarsePaths:

        transition_tree = self.build_transition_tree(
            goal_distribution, np.array(past_trajectory.iloc[-1][["x", "y", "z"]])
        )

        coarse_paths = CoarsePaths(transition_tree=transition_tree)
        return coarse_paths

    def on_episode_init(self, scene_graph: DSG):
        self.scene_graph = scene_graph
        self.places_layer_nx = construct_places_layer_nx(scene_graph)
        node_positions = [
            node[1]["position"] for node in self.places_layer_nx.nodes(data=True)
        ]
        self.kdtree = cKDTree(node_positions)

    def build_transition_tree(
        self, goal_distribution: GoalSequences, start_position: list[float]
    ) -> Tree:
        """
        Build the transition tree from the instance sequences tree.
        Instance sequences tree has shortest paths saved in edge attributes. Transition tree operates directly on places layer.
        Transition tree is a directed graph with nodes of type INCOMING, OUTGOING, PASSING
        Transition tree can be used to build Q-matrix for CTMC
        """
        interaction_sequences_tree = goal_distribution.tree.data
        root_id = goal_distribution.tree.root
        current_interaction_time_mean = goal_distribution.tree.data.nodes[root_id][
            "duration"
        ]
        transition_tree, id_map, node_id = self.initialize_transition_tree(
            interaction_sequences_tree,
            start_position,
            current_interaction_time_mean,
            root_id,
        )

        transition_tree, id_map, node_id = self.build_transition_tree_recursive(
            interaction_sequences_tree, transition_tree, id_map, node_id, root_id
        )

        return Tree(data=transition_tree, root=root_id)

    def build_transition_tree_recursive(
        self,
        interaction_sequences_tree: nx.DiGraph,
        transition_tree: nx.DiGraph,
        id_map: dict,
        node_id: int,
        parent_id: int,
    ) -> None:
        for next_node in interaction_sequences_tree.neighbors(parent_id):
            edge = (parent_id, next_node)
            edge_probability = interaction_sequences_tree.edges[edge]["probability"]
            path = interaction_sequences_tree.edges[edge]["places_path"]
            for path_idx, places_node in enumerate(path[:-1]):
                if path_idx == 0:
                    transition_tree.add_node(
                        node_id,
                        type=NodeTypesTransitionTree.PROBABILITY,
                        places_node_id=places_node,
                        position=interaction_sequences_tree.nodes[parent_id][
                            "position"
                        ],
                    )
                    transition_tree.add_edge(
                        id_map[edge[0]]["outgoing"],
                        node_id,
                        transition_rate=edge_probability,
                        edge_type="interaction_probability",
                    )
                    prob_node = node_id
                    node_id += 1
                    transition_tree.add_node(
                        node_id,
                        type=NodeTypesTransitionTree.PASSING,
                        places_node_id=places_node,
                        position=self.places_layer_nx.nodes[places_node]["position"],
                    )
                    edge_distance = euclidean_distance(
                        interaction_sequences_tree.nodes[parent_id]["position"],
                        self.places_layer_nx.nodes[places_node]["position"],
                    )
                    transition_rate = self.calculate_transition_rate(
                        edge_distance / self.walking_speed
                    )
                    transition_tree.add_edge(
                        prob_node, node_id, transition_rate=transition_rate
                    )
                else:
                    transition_tree.add_node(
                        node_id,
                        type=NodeTypesTransitionTree.PASSING,
                        places_node_id=places_node,
                        position=self.places_layer_nx.nodes[places_node]["position"],
                    )
                    edge_distance = self.places_layer_nx.edges[
                        path[path_idx - 1], places_node
                    ]["weight"]
                    transition_rate = self.calculate_transition_rate(
                        edge_distance / self.walking_speed
                    )
                    transition_tree.add_edge(
                        node_id - 1, node_id, transition_rate=transition_rate
                    )
                node_id += 1
            if len(path) <= 1:
                places_node = interaction_sequences_tree.nodes[next_node][
                    "places_node_id"
                ]
                transition_tree.add_node(
                    node_id,
                    type=NodeTypesTransitionTree.PASSING,
                    places_node_id=places_node,
                    position=self.places_layer_nx.nodes[places_node]["position"],
                )
                transition_tree.add_edge(
                    id_map[edge[0]]["outgoing"],
                    node_id,
                    transition_rate=edge_probability,
                    edge_type="interaction_probability",
                )
                node_id += 1
                last_path_node_id = node_id - 1
                edge_distance = 0.0
            else:
                last_path_node_id = node_id - 1
                edge_distance = self.places_layer_nx.edges[path[-2], path[-1]]["weight"]
            node_attr = interaction_sequences_tree.nodes[next_node]
            position = self.places_layer_nx.nodes[node_attr["places_node_id"]][
                "position"
            ]
            incoming = node_id
            node_id += 1
            outgoing = node_id
            node_id += 1
            transition_tree.add_node(
                incoming, type=NodeTypesTransitionTree.INCOMING, **node_attr
            )
            transition_tree.add_edge(
                last_path_node_id,
                incoming,
                transition_rate=self.calculate_transition_rate(
                    edge_distance / self.walking_speed
                ),
            )
            transition_tree.add_node(
                outgoing, type=NodeTypesTransitionTree.OUTGOING, **node_attr
            )
            transition_tree.add_edge(
                incoming,
                outgoing,
                transition_rate=self.calculate_transition_rate(node_attr["duration"]),
            )
            id_map[next_node] = {"incoming": incoming, "outgoing": outgoing}

            transition_tree, id_map, node_id = self.build_transition_tree_recursive(
                interaction_sequences_tree, transition_tree, id_map, node_id, next_node
            )

        return transition_tree, id_map, node_id

    def initialize_transition_tree(
        self,
        interaction_sequences_tree: nx.DiGraph,
        start_position: list[float],
        current_interaction_time_mean: float,
        root_id: int,
    ) -> tuple[nx.DiGraph, dict, int]:
        """initializes transition tree with root node"""
        transition_tree = nx.DiGraph()
        id_map = {}
        node_id = root_id
        start_node = interaction_sequences_tree.nodes[root_id]
        transition_tree.add_node(
            root_id, type=NodeTypesTransitionTree.INCOMING, **start_node
        )
        node_id += 1
        transition_tree.add_node(
            node_id, type=NodeTypesTransitionTree.OUTGOING, **start_node
        )
        transition_rate = self.calculate_transition_rate(current_interaction_time_mean)
        transition_tree.add_edge(root_id, node_id, transition_rate=transition_rate)
        id_map[root_id] = {"incoming": root_id, "outgoing": node_id}
        node_id += 1
        return transition_tree, id_map, node_id

    def calculate_transition_rate(self, duration):
        """Calculates the transition rate based on duration."""
        if duration and round(duration, 4) > 0:
            return 1 / duration
        else:
            return 1 / self.DEFAULT_TIME
