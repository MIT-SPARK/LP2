import numpy as np
import os
from typing import List, Tuple, Union
import pandas as pd
import logging
import networkx as nx
from collections import Counter
import json
from scipy.spatial import cKDTree
from enum import IntEnum, unique
import ast

import spark_dsg as dsg
import spark_dsg.networkx as dsg_nx
from spark_dsg._dsg_bindings import NodeSymbol, DynamicSceneGraph as DSG

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), organization=os.getenv("OPENAI_API_ORG")
)

from lhmp.utils.dsg import Layers, get_closest_node, construct_places_layer_nx
from lhmp.utils.language import join_to_sentence
from lhmp.utils.general import read_json, performance_measure
from lhmp.utils.visualization import visualize_interaction_prediction_tree
from lhmp.utils.method import NodeTypesTransitionTree
from lhmp.utils.data import GoalSequences, Tree
from lhmp.utils.math import euclidean_distance

from lhmp.modules.goal.goal_module_base import GoalModuleBase
from lhmp.modules.goal.llm import LlmGoalModule


class LlmGoalModuleLP2(LlmGoalModule):
    """
    inputs: self, scene_graph: DSG, room_labels: dict, previous_interacions: list[dict], past_trajectory: list[dict]
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if "depth" in kwargs:
            self.depth = kwargs["depth"]
        else:
            self.depth = 2

        if "width" in kwargs:
            self.width = kwargs["width"]
        else:
            self.width = 2

        if "avg_walking_speed" in kwargs:
            self.walking_speed = kwargs["avg_walking_speed"]
        else:
            self.walking_speed = 1.4

        if "num_instances_per_class" in kwargs:
            self.num_instances_per_class = kwargs["num_instances_per_class"]
        else:
            self.num_instances_per_class = 5

        assert (
            "example" in kwargs["llm_prompts"]
        ), "llm prompt textfiles not specified in kwargs"
        self.example_path = kwargs["llm_prompts"]["example"]

        assert (
            "current_interaction" in kwargs["llm_prompts"]
        ), "llm prompt textfiles not specified in kwargs"
        self.current_interaction_path = kwargs["llm_prompts"]["current_interaction"]

        assert "include_actions" in kwargs, "include_actions not specified in kwargs"
        self.include_actions = kwargs["include_actions"]

        with open(self.example_path, "r") as f:
            self.example_text = f.read()

        with open(self.instructions_path, "r") as f:
            self.instructions = f.read()

        with open(self.current_interaction_path, "r") as f:
            self.current_interaction_instructions = f.read()

        self.instructions = self.instructions.replace("</width>", str(self.width))

        self.node_index_semantic = None
        self.node_index_instance = None
        self.interaction_sequences_tree = None
        self.interaction_sequences_semantics_tree = None
        self.root_id_semantic = None
        self.root_id_instance = None
        self.closest_nodes = None

        self.DEFAULT_TIME = 0.1  # [s]

        self.scene_graph = None
        self.places_layer_nx = None
        self.shortest_paths_between_objects = None  # shortest paths between objects. call as self.shortest_paths_between_objects[source][target]["path"] or ["distance"]
        self.kdtree_places = None

        self.__depth = None

        self.error_addition = """\nThe last time you answered this prompt, there was an error subsequently. Make sure the output is syntactically correct and all objects exist in the environment."""

    def on_episode_init(self, scene_graph: DSG, room_labels: dict) -> None:
        self.scene_graph = scene_graph
        self.places_layer_nx = construct_places_layer_nx(scene_graph)

        node_positions = [
            node[1]["position"] for node in self.places_layer_nx.nodes(data=True)
        ]
        self.kdtree_places = cKDTree(node_positions)
        objects_layer_nx = dsg_nx.layer_to_networkx(
            scene_graph.get_layer(Layers.OBJECTS)
        )
        self.objects_node_information, self.objects_by_semantic_class = (
            self.get_objects_node_information(objects_layer_nx)
        )

        with performance_measure("shortest paths"):
            self.shortest_paths_between_objects = (
                self.get_shortest_paths_between_objects()
            )

    def find_goals(
        self,
        scene_graph: DSG,
        room_labels: dict,
        previous_interactions: list[dict],
        past_trajectory: pd.DataFrame,
    ) -> GoalSequences:

        current_position = list(past_trajectory.iloc[-1][["x", "y", "z"]])
        self.reset_prediction(current_position)
        closest_current_places_node, _ = get_closest_node(
            self.places_layer_nx, current_position, self.kdtree_places
        )
        previous_interactions_semantics = self.get_previous_interactions_semantics(
            previous_interactions, room_labels
        )
        graph_query = self.graph_to_query_spark_withrooms(room_labels)
        self.predict_goals_recursive(
            graph_query, previous_interactions_semantics, parent_id=self.root_id
        )
        if past_trajectory.iloc[-1]["interaction_id"] != -1:
            current_interaction_time_prediction = self.predict_current_interaction_time(
                previous_interactions, past_trajectory
            )
            if current_interaction_time_prediction < self.DEFAULT_TIME:
                current_interaction_time_prediction = self.DEFAULT_TIME
        else:
            current_interaction_time_prediction = self.DEFAULT_TIME
        self.root_id_instance = self.add_node_to_instance_tree(
            node_id=self.root_id,
            position=current_position,
            places_node_id=closest_current_places_node,
            depth=0,
            room_id=self.places_layer_nx.nodes[closest_current_places_node]["room_id"],
            probability=1.0,
            duration=current_interaction_time_prediction,
        )
        self.build_instance_sequences_tree(current_position)

        goal_sequences_tree = GoalSequences(
            tree=Tree(data=self.interaction_sequences_tree, root=self.root_id_instance)
        )

        return goal_sequences_tree

    def reset_prediction(self, current_position: list[float]) -> None:
        self.node_index_semantic = 0
        self.node_index_instance = 0
        self.__depth = 0
        self.closest_nodes = self.get_closest_nodes(current_position)
        self.interaction_sequences_tree = nx.DiGraph()
        self.interaction_sequences_semantics_tree = nx.DiGraph()
        self.root_id = self.add_node_to_semantic_tree()

    def get_closest_nodes(self, root_position) -> dict:

        closest_node, distance_to_closest = get_closest_node(
            self.places_layer_nx, root_position
        )
        distances = {
            i: euclidean_distance(
                root_position, self.places_layer_nx.nodes[i]["position"]
            )
            for i in self.places_layer_nx.nodes
        }
        closest_nodes = [closest_node] + [
            i
            for i in self.places_layer_nx.nodes
            if i != closest_node and distances[i] <= 0.5
        ]
        closest_nodes = {
            i: euclidean_distance(
                root_position, self.places_layer_nx.nodes[i]["position"]
            )
            for i in closest_nodes
        }

        return closest_nodes

    def predict_goals_recursive(
        self, graph_query: str, previous_interactions: list[dict], parent_id: int
    ) -> None:
        """
        Recursively predict goals until the depth is reached
        """
        if self.__depth == self.depth:
            return
        else:
            self.__depth += 1
            print("new depth: ", self.__depth)
            try:
                goals = self.predict_next_goals(
                    graph_query,
                    previous_interactions
                    + self.get_previous_predicted_interactions(parent_id),
                )
                for goal in goals:
                    if self.include_actions:
                        node_id = self.add_node_to_semantic_tree(
                            depth=self.__depth,
                            parent_index=parent_id,
                            semantic_label=goal["object"],
                            probability=goal["probability"],
                            duration=goal["duration"],
                            action=goal["action"],
                        )
                    else:
                        node_id = self.add_node_to_semantic_tree(
                            depth=self.__depth,
                            parent_index=parent_id,
                            semantic_label=goal["object"],
                            probability=goal["probability"],
                            duration=goal["duration"],
                        )
                    self.predict_goals_recursive(
                        graph_query, previous_interactions, node_id
                    )
            except Exception as e:
                print("syntax error: ", e)
                logging.error(f"syntax error: {e}\n retrying with error prompt.")
                goals = self.predict_next_goals(
                    graph_query,
                    previous_interactions
                    + self.get_previous_predicted_interactions(parent_id),
                    error_addition=self.error_addition,
                )
                for goal in goals:
                    node_id = self.add_node_to_semantic_tree(
                        depth=self.__depth,
                        parent_index=parent_id,
                        semantic_label=goal["object"],
                        probability=goal["probability"],
                        duration=goal["duration"],
                    )
                    self.predict_goals_recursive(
                        graph_query, previous_interactions, node_id
                    )
            self.__depth -= 1

    def get_previous_predicted_interactions(self, parent_id: int) -> list[dict]:
        """
        Get the previous predicted interactions from the semantic tree
        """
        _parent_id = parent_id
        pred_interactions = []
        while True:
            if _parent_id is self.root_id:
                break
            else:
                node = self.interaction_sequences_semantics_tree.nodes[_parent_id]
                if self.include_actions:
                    attr = {
                        "semantic_label": node["semantic_label"],
                        "action": node["action"],
                        "probability": node["probability"],
                        "duration": node["duration"],
                    }
                else:
                    attr = {
                        "semantic_label": node["semantic_label"],
                        "probability": node["probability"],
                        "duration": node["duration"],
                    }
                pred_interactions.insert(0, attr)
                _parent_id = next(
                    self.interaction_sequences_semantics_tree.predecessors(_parent_id)
                )

        return pred_interactions

    def add_node_to_semantic_tree(
        self,
        depth: int = 0,
        parent_index: int = None,
        semantic_label: str = None,
        probability: float = None,
        duration: float = None,
        action: str = None,
    ) -> None:
        node_id = self.node_index_semantic
        node_attr = {
            "depth": depth,
            "parent": parent_index,
            "semantic_label": semantic_label,
            "probability": probability,
            "duration": duration,
        }
        if action is not None:
            node_attr["action"] = action
        self.interaction_sequences_semantics_tree.add_node(
            self.node_index_semantic, **node_attr
        )
        if parent_index is not None:
            self.interaction_sequences_semantics_tree.add_edge(
                parent_index, self.node_index_semantic
            )
        self.node_index_semantic += 1
        return node_id

    def add_node_to_instance_tree(
        self,
        node_id: int,
        position: list[float],
        places_node_id: int,
        depth: int,
        parent_index: int = None,
        semantic_label: str = None,
        room_id: int = -1,
        probability: float = None,
        duration: float = None,
    ) -> None:
        node_id_instance = self.node_index_instance
        node_attr = {
            "id": node_id,
            "sg_id": node_id,
            "position": position,
            "places_node_id": places_node_id,
            "semantic_label": semantic_label,
            "room_id": room_id,
            "probability": probability,
            "duration": duration,
            "depth": depth,
        }
        self.interaction_sequences_tree.add_node(node_id_instance, **node_attr)

        if parent_index is not None:
            if parent_index == self.root_id_instance:
                places_path, distance = self.get_path_from_root(places_node_id)
            else:
                parent_places_node = self.interaction_sequences_tree.nodes[
                    parent_index
                ]["places_node_id"]
                places_path, distance = self.get_path_between_objects(
                    places_node_id, parent_places_node
                )
            edge_attr = {
                "probability": probability,
                "places_path": places_path,
                "distance": distance,
            }
            self.interaction_sequences_tree.add_edge(
                parent_index, node_id_instance, **edge_attr
            )

        self.node_index_instance += 1

        return node_id_instance

    def get_path_from_root(self, places_node_id) -> tuple[list[int], float]:

        min_distance = np.inf
        places_path = None
        for node, distance in self.closest_nodes.items():
            node_places_path = self.shortest_paths_between_objects[places_node_id][
                node
            ]["path"][::-1]
            path_distance = (
                self.shortest_paths_between_objects[places_node_id][node]["distance"]
                + distance
            )
            if path_distance < min_distance:
                min_distance = path_distance
                places_path = node_places_path

        return places_path, min_distance

    def get_path_between_objects(
        self, places_node_id: int, parent_places_node: int
    ) -> tuple[list[int], float]:

        if parent_places_node not in self.shortest_paths_between_objects:
            places_path = self.shortest_paths_between_objects[places_node_id][
                parent_places_node
            ]["path"][::-1]
            distance = self.shortest_paths_between_objects[places_node_id][
                parent_places_node
            ]["distance"]
        else:
            places_path = self.shortest_paths_between_objects[parent_places_node][
                places_node_id
            ]["path"]
            distance = self.shortest_paths_between_objects[parent_places_node][
                places_node_id
            ]["distance"]

        return places_path, distance

    def predict_next_goals(
        self,
        graph_query: str,
        previous_interactions: list[dict],
        error_addition: str = "",
    ) -> dict:
        logging.info(f"width: {self.width}\n")
        logging.info(f"depth: {self.depth - self.__depth}\n")
        previous_interactions_query = self.previous_interactions_to_query(
            previous_interactions
        )
        logging.info(f"previous_interactions: \n'{previous_interactions_query}'\n")
        prompt = graph_query + previous_interactions_query + self.instructions
        if error_addition:
            prompt += error_addition
        response = self.query_llm(prompt)
        logging.info(f"LLM response: \n'{response.choices[0].message.content}'\n")
        llm_result = self.postprocess_response(response)
        llm_result = self.error_handler(
            llm_result, prompt, response.choices[0].message.content
        )
        return llm_result

    def error_handler(self, llm_result: list[dict], prompt: str, response_content: str):
        nonexisting_objects = []
        for pred in llm_result:
            if pred["object"] not in self.objects_by_semantic_class:
                if (
                    pred["object"].endswith("s")
                    and pred["object"][:-1] in self.objects_by_semantic_class
                ):  # LLM may predict plural of object
                    pred["object"] = pred["object"][:-1]
                elif pred["object"] + "s" in self.objects_by_semantic_class:
                    pred["object"] = pred["object"] + "s"
                else:
                    nonexisting_objects.append(pred["object"])
                    break
        if not nonexisting_objects:
            return llm_result
        else:
            logging.warning(
                f"LLM predicted nonexisting objects: {', '.join(nonexisting_objects)}, retrying with error handler."
            )
            print(
                f"LLM predicted nonexisting object: {', '.join(nonexisting_objects)}, retrying with error handler."
            )
            answer = {"role": "system", "content": response_content}
            new_prompt = {
                "role": "user",
                "content": f"the objects [{', '.join(nonexisting_objects)}] do not exist in the environment. Make sure that all objects exist in the provided environment. Output your answer in the same JSON format.",
            }
            chat_prompt = [answer, new_prompt]
            new_response = self.query_llm(prompt, chat_prompt)
            logging.info(
                f"new LLM response: \n'{new_response.choices[0].message.content}'\n"
            )
            llm_result = self.postprocess_response(new_response)
            llm_result = self.error_handler(
                llm_result, prompt, new_response.choices[0].message.content
            )
            return llm_result

    def test_if_in_environment(self, semantic_class: str) -> Union[str, bool]:
        """
        Test if the semantic class is in the environment
        """
        if semantic_class in self.objects_by_semantic_class:
            return semantic_class
        else:
            if (
                semantic_class.endswith("s")
                and semantic_class[:-1] in self.objects_by_semantic_class
            ):  # LLM may predict plural of object
                semantic_class = semantic_class[:-1]
                return semantic_class
            else:
                return ""

    def previous_interactions_to_query(self, previous_interactions: list[dict]) -> str:
        if self.include_actions:
            return self.previous_interactions_to_query_with_actions(
                previous_interactions
            )
        else:
            return self.previous_interactions_to_query_without_actions(
                previous_interactions
            )

    def previous_interactions_to_query_without_actions(
        self, previous_interactions: list[dict]
    ) -> str:
        previous_interactions_string = join_to_sentence(
            [
                f"a {i['semantic_label']} for {i['duration']:.2f} seconds"
                for i in previous_interactions
            ]
        )
        with open(self.previous_interactions_path, "r") as f:
            text = f.read()
        text = text.replace(
            "</previous_interactions_string>", previous_interactions_string
        )
        return text

    def previous_interactions_to_query_with_actions(
        self, previous_interactions: list[dict]
    ) -> str:
        previous_interactions_string = join_to_sentence(
            [
                f"'{i['action']}' with {i['semantic_label']} for {i['duration']:.2f} seconds"
                for i in previous_interactions
            ]
        )
        with open(self.previous_interactions_path, "r") as f:
            text = f.read()
        text = text.replace(
            "</previous_interactions_string>", previous_interactions_string
        )
        return text

    def query_llm(self, prompt: str, error_addition: list[dict] = []) -> ChatCompletion:
        if error_addition:
            chat_prompt = [
                {
                    "role": "system",
                    "content": "you are backend prediction model designed to output JSON.",
                },
                {"role": "user", "content": prompt},
            ] + error_addition
        else:
            chat_prompt = [
                {
                    "role": "system",
                    "content": "you are backend prediction model designed to output JSON.",
                },
                {"role": "user", "content": prompt},
            ]
        response = client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=chat_prompt,
            temperature=self.temperature,
        )

        return response

    def postprocess_response(self, response: ChatCompletion) -> dict:
        """
        Postprocess the response from the OpenAI API from dict of strings to list of tuples
        """
        try:
            response_json = json.loads(response.choices[0].message.content)
            reasoning = response_json["reasoning"]
            prediction_list = response_json["predictions"]
        except Exception as e:
            print(e)
            SyntaxError("Could not parse prediction from OpenAI API")

        prediction_list = self.normalize_probabilities(prediction_list)

        return prediction_list

    def normalize_probabilities(self, llm_result: list[dict]) -> list[dict]:
        """
        Normalize the probabilities of the LLM result
        """
        total_probability = sum([i["probability"] for i in llm_result])
        for i in llm_result:
            i["probability"] = i["probability"] / total_probability
        return llm_result

    def append_goal_to_previous_interactions(
        self, goal: tuple[dict, float, float], previous_interactions: list[dict]
    ) -> None:
        """
        Append a goal to the previous interactions
        """

        new_previous_interactions = previous_interactions + [
            {
                "semantic_label": goal["object"],
                "probability": goal["probability"]
                * np.prod(
                    [
                        goal["probability"]
                        for goal in previous_interactions
                        if "probability" in goal
                    ]
                ),
                "duration": goal["duration"],
            }
        ]

        return new_previous_interactions

    def get_objects_node_information(self, objects_layer_nx: nx.Graph) -> list[dict]:
        """
        Get the information of the nodes in the objects layer
        """
        objects_node_information = {}
        objects_by_semantic_class = {}
        for node in objects_layer_nx.nodes:
            node_information = self.get_node_information(node)
            objects_node_information[node] = node_information
            semantic_class = objects_layer_nx.nodes[node]["name"]
            if semantic_class not in objects_by_semantic_class:
                objects_by_semantic_class[semantic_class] = []
            objects_by_semantic_class[semantic_class].append(node_information)
        return objects_node_information, objects_by_semantic_class

    def get_node_information(self, node: int) -> dict:
        """
        Get the information of a node in the objects layer
        """
        node_information = {}
        node_information["node_id"] = NodeSymbol(node).category_id
        node_information["semantic_class"] = self.scene_graph.get_node(
            node
        ).attributes.name
        node_information["position"] = self.scene_graph.get_node(
            node
        ).attributes.position
        (
            node_information["nearest_places_node"],
            node_information["distance_to_nearest_places_node"],
        ) = get_closest_node(
            self.places_layer_nx, node_information["position"], self.kdtree_places
        )

        room_id = [
            NodeSymbol(i.source).category_id
            for i in self.scene_graph.interlayer_edges
            if i.target == node
        ]
        if room_id:
            room_id = room_id[0]
        else:
            room_id = -1
        node_information["room_id"] = room_id
        return node_information

    def get_shortest_paths_between_objects(self) -> dict:
        """
        Get the shortest paths between objects
        """
        shortest_paths_between_objects = {}
        for source_attr in self.objects_node_information.values():
            shortest_paths_between_objects[source_attr["nearest_places_node"]] = {}
            distances, paths = nx.single_source_dijkstra(
                self.places_layer_nx,
                source_attr["nearest_places_node"],
                weight="weight",
            )
            for key, dist in distances.items():
                shortest_paths_between_objects[source_attr["nearest_places_node"]][
                    key
                ] = {"path": paths[key], "distance": dist}
        return shortest_paths_between_objects

    def predict_current_interaction_time(
        self,
        previous_interactions: list[dict],
        past_trajectory: pd.DataFrame,
        error_str: str = None,
    ) -> None:

        if "action" in previous_interactions[0]:
            previous_interactions_string = join_to_sentence(
                [
                    f"'{i['action']}' with {i['semantic_label']} for {i['duration']:.2f} seconds"
                    for i in previous_interactions
                ]
            )
        else:
            previous_interactions_string = "\n".join(
                [
                    f"{i['semantic_label']} for {i['duration']:.2f} seconds"
                    for i in previous_interactions
                ]
            )
        last_interaction_sem = previous_interactions[-1]["semantic_label"]
        last_interaction_duration = previous_interactions[-1]["duration"]
        prompt = self.current_interaction_instructions.replace(
            "</previous_interactions_string>", previous_interactions_string
        )
        prompt = prompt.replace("</last_interaction_sem>", last_interaction_sem)
        prompt = prompt.replace(
            "</last_interaction_duration>", f"{last_interaction_duration:.2f}"
        )
        if error_str:
            prompt += error_str
        response = self.query_llm(prompt)
        response_json = json.loads(response.choices[0].message.content)
        logging.info(f"current interaction prompt: \n {prompt}\n")
        logging.info(
            f"current interaction answer: \n{response_json['reasoning']}\nduration:: {response_json['duration']}"
        )
        try:
            if response_json["duration"] == None:
                raise Exception("duration is None")
            if response_json["duration"].__class__ == str:
                current_interaction_time = ast.literal_eval(response_json["duration"])
            else:
                print("GPT output is float")
                current_interaction_time = response_json["duration"]
        except Exception as e:
            print("current interaction time prediction error: ", e)
            error_str = f"there was an error in the previous call of the LLM in the backend pipeline: {e}\nmake sure the output 'duration' is of type float."
            current_interaction_time = self.predict_current_interaction_time(
                previous_interactions, past_trajectory, error_str
            )
        return current_interaction_time

    def build_instance_sequences_tree(self, position: list[float]) -> None:
        """
        Build the instance sequences tree from the semantic sequences tree
        TODO: extend to recursive setting, where the instance sequences tree is built from the semantic sequences tree
        """
        nearest_previous_places_node, _ = get_closest_node(
            self.places_layer_nx, position, self.kdtree_places
        )

        self.build_instance_sequences_tree_recursive(
            nearest_previous_places_node,
            parent_id_semantic=self.root_id,
            parent_id_instance=self.root_id,
        )

    def build_instance_sequences_tree_recursive(
        self,
        nearest_previous_places_node: int,
        parent_id_semantic: int,
        parent_id_instance: int,
    ) -> None:
        """
        Build the instance sequences tree from the semantic sequences tree
        """
        for node in self.interaction_sequences_semantics_tree.successors(
            parent_id_semantic
        ):
            node_attr = self.interaction_sequences_semantics_tree.nodes[node]
            semantic_class = node_attr["semantic_label"]
            probability_semantic = node_attr["probability"]
            duration = node_attr["duration"]
            depth = node_attr["depth"]

            # if semantic_class == parent_semantic_class:
            #     continue

            object_instances = self.objects_by_semantic_class[semantic_class].copy()
            object_instances = self.compute_instance_probabilities(
                nearest_previous_places_node, object_instances, probability_semantic
            )

            if len(object_instances) > self.num_instances_per_class:
                object_instances = sorted(
                    object_instances, key=lambda i: i["probability"], reverse=True
                )[: self.num_instances_per_class]

                # renormalizing
                sum_probabilities = sum([i["probability"] for i in object_instances])
                for instance in object_instances:
                    instance["probability"] = (
                        instance["probability"]
                        * probability_semantic
                        / sum_probabilities
                    )

            for instance in object_instances:
                instance_id = self.add_node_to_instance_tree(
                    node_id=instance["node_id"],
                    position=self.places_layer_nx.nodes[
                        instance["nearest_places_node"]
                    ]["position"],
                    places_node_id=instance["nearest_places_node"],
                    depth=depth,
                    parent_index=parent_id_instance,
                    semantic_label=semantic_class,
                    room_id=instance["room_id"],
                    probability=instance["probability"],
                    duration=duration,
                )

                self.build_instance_sequences_tree_recursive(
                    instance["nearest_places_node"], node, instance_id
                )

    def compute_instance_probabilities(
        self,
        nearest_previous_places_node: int,
        object_instances: list[dict],
        semantic_probability: float,
    ) -> list[dict]:
        """
        Compute the instance probabilities of the predicted interaction sequences
        """
        distances = []
        for idx, instance in enumerate(object_instances):
            if nearest_previous_places_node not in self.shortest_paths_between_objects:
                distances.append(
                    self.shortest_paths_between_objects[
                        instance["nearest_places_node"]
                    ][nearest_previous_places_node]["distance"]
                )
            else:
                distances.append(
                    self.shortest_paths_between_objects[nearest_previous_places_node][
                        instance["nearest_places_node"]
                    ]["distance"]
                )

        distances = np.array(distances)
        if np.sum(distances) == 0:
            distances = np.ones_like(distances)

        # probabilites inversely proportional to distance
        probabilities = (max(distances) + 1.0 - distances) / sum(
            max(distances) + 1.0 - distances
        )

        probabilities *= semantic_probability

        for idx, instance in enumerate(object_instances):
            instance["probability"] = probabilities[idx]

        return object_instances

    def get_previous_interactions_semantics(
        self, previous_interactions: list[dict], room_labels: dict
    ) -> list[dict]:
        if self.include_actions:
            return [
                {
                    "semantic_label": i["semantic_label"],
                    "action": i["action"],
                    "room": room_labels[f"R({i['room_id']})"],
                    "duration": i["duration"],
                }
                for i in previous_interactions
            ]
        else:
            return [
                {
                    "semantic_label": i["semantic_label"],
                    "room": room_labels[f"R({i['room_id']})"],
                    "duration": i["duration"],
                }
                for i in previous_interactions
            ]

    def graph_to_query_spark_withrooms(self, room_labels: dict) -> str:

        room_nodes = list(self.scene_graph.get_layer(Layers.ROOMS).nodes)
        room_edges = list(self.scene_graph.get_layer(Layers.ROOMS).edges)

        room_str_list: list[str] = []
        room_edges_semantics: list[str] = {}

        for node in room_nodes:
            room_str_list.append(room_labels[node.attributes.name])
            connected_nodes = [
                NodeSymbol(edge.target).category_id
                for edge in room_edges
                if node.id.value == edge.source
            ] + [
                NodeSymbol(edge.source).category_id
                for edge in room_edges
                if node.id.value == edge.target
            ]
            room_edges_semantics[room_labels[node.attributes.name]] = [
                room_labels[f"R({i})"] for i in connected_nodes
            ]

        room_string = join_to_sentence(room_str_list)
        room_edges_strings = {
            k: join_to_sentence(v) for k, v in room_edges_semantics.items()
        }
        room_to_room_edges = "\n".join(
            [
                f"From {k} you can directly access {v}"
                for k, v in room_edges_strings.items()
            ]
        )
        objects_in_room = self.get_objects_in_room(room_labels)
        objects_in_room_help_dict = {
            k: join_to_sentence(v) for k, v in objects_in_room.items()
        }
        objects_in_room_string = "\n".join(
            [
                f"In {room_labels[k]} there is {v}"
                for k, v in objects_in_room_help_dict.items()
            ]
        )

        with open(self.environment_path) as f:
            text = f.read()
        text = text.replace("</room_string>", room_string)
        text = text.replace("</room_to_room_edges>", room_to_room_edges)
        text = text.replace("</objects_in_room_string>", objects_in_room_string)
        return text

    def get_objects_in_room(self, room_labels: dict) -> dict[str, int]:
        objects_in_room = {}
        for room, room_label in room_labels.items():
            room_id = int(room.split("(")[1].split(")")[0])
            object_ids_in_room = [
                i
                for i in self.scene_graph.get_node(
                    NodeSymbol("R", room_id).value
                ).children()
                if NodeSymbol(i).category == "O"
            ]

            object_list = []
            for obj_node in object_ids_in_room:
                obj_sem_class = self.scene_graph.get_node(obj_node).attributes.name
                object_list.append(obj_sem_class)

            if object_list:
                objects_in_room[room] = Counter(object_list)

        return objects_in_room
