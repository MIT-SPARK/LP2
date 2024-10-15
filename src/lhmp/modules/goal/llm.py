import numpy as np
import os
from typing import List, Tuple, Union
import pandas as pd

import spark_dsg as dsg
from spark_dsg._dsg_bindings import NodeSymbol, DynamicSceneGraph as DSG

import openai
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), organization=os.getenv("OPENAI_API_ORG")
)
import ast

from lhmp.utils.dsg import Layers
from lhmp.modules.goal.goal_module_base import GoalModuleBase
from lhmp.utils.general import read_json


### TODO:
# - first results not encouraging at all (gpt predicts bogus sequences)
# - could do per-node classification for [precondition, action, effects]
# - need to predict interaction times (per object); for now this could be set to a constant;
# - only predict spatially and assume constant velocity


class LlmGoalModule(GoalModuleBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from dotenv import load_dotenv

        load_dotenv()
        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
        else:
            self.temperature = 0.0

        assert "llm_prompts" in kwargs, "llm prompt textfiles not specified in kwargs"
        assert (
            "environment" in kwargs["llm_prompts"]
        ), "environment prompt textfile not specified in kwargs"

        if "include_actions" in kwargs:
            self.include_actions = kwargs["include_actions"]
        else:
            self.include_actions = False

        if self.include_actions:
            assert (
                "previous_interactions_with_actions" in kwargs["llm_prompts"]
            ), "llm prompt textfiles not specified in kwargs"
            assert (
                "instructions_with_actions" in kwargs["llm_prompts"]
            ), "instructions prompt textfile not specified in kwargs"
            self.previous_interactions_path = kwargs["llm_prompts"][
                "previous_interactions_with_actions"
            ]
            self.instructions_path = kwargs["llm_prompts"]["instructions_with_actions"]
        else:
            assert (
                "previous_interactions" in kwargs["llm_prompts"]
            ), "previous_interactions prompt textfile not specified in kwargs"
            assert (
                "instructions" in kwargs["llm_prompts"]
            ), "instructions prompt textfile not specified in kwargs"
            self.instructions_path = kwargs["llm_prompts"]["instructions"]
            self.previous_interactions_path = kwargs["llm_prompts"][
                "previous_interactions"
            ]

        self.environment_path = kwargs["llm_prompts"]["environment"]
        self.model = kwargs["gpt_model"]

        self.error_addition = """\nThe last time you answered this prompt, there was an error subsequently. Make sure the output is syntactically correct and all node ids exist in the environment."""

    def find_goals(
        self,
        scene_graph: DSG,
        room_labels: dict,
        previous_interactions: list[dict],
        past_trajectory: pd.DataFrame,
    ) -> list[list[tuple[dict, float, float]]]:

        prompt = self.combine_prompt(scene_graph, previous_interactions, room_labels)
        response = self.query_llm(prompt)
        llm_result = self.postprocess_response(response)
        return llm_result

    def query_llm(self, prompt: str) -> list[list[tuple[dict, float]]]:
        chat_prompt = [
            {
                "role": "system",
                "content": "YouAre:BackendPredictionModel,PredictLongTermHumanActionSequences,QueriedAutoregressively,APISchemaCompliant",
            },
            {"role": "user", "content": prompt},
        ]
        response = client.chat.completions.create(
            model=self.model, messages=chat_prompt, temperature=self.temperature
        )

        return response

    def postprocess_response(self, response: dict) -> list[list[tuple[dict, float]]]:
        """
        Postprocess the response from the OpenAI API from dict of strings to list of tuples
        """
        prediction = response["choices"][0]["message"]["content"]

        # TODO: might need to add further conditional postprocessing here to improve robustness to faulty responses
        if prediction.endswith("."):
            prediction = prediction[:-1]

        try:
            prediction_list = ast.literal_eval(prediction.replace(" -> ", ", "))
        except Exception as e:
            print(e)
            SyntaxError("Could not parse prediction from OpenAI API")

        # TODO: move to new data structure

        return prediction_list

    def combine_prompt(
        self, scene_graph: DSG, previous_interactions: list[dict], room_labels: dict
    ) -> str:
        graph_query = self.graph_to_query_spark(scene_graph, room_labels)
        previous_interactions_query = self.previous_interactions_to_query(
            previous_interactions
        )
        prompt = graph_query + previous_interactions_query
        with open(self.instructions_path, "r") as f:
            prompt += f.read()
        return prompt

    def previous_interactions_to_query(self, interactions: list[dict]) -> str:
        """
        Convert a previous interaction into a query for the OpenAI API
        """
        previous_interactions_str = (
            "["
            + " -> ".join(
                [
                    f"('id': {i['id']}, 'semantic_label': {i['semantic_label']}, 'room_id': {i['room_id']}, 'duration': {i['duration']})"
                    for i in interactions
                ]
            )
            + "]"
        )
        with open(self.previous_interactions_path, "r") as f:
            text = f.read()
        text = text.replace("</previous_interactions_str>", previous_interactions_str)
        return text

    def graph_to_query_spark(self, scene_graph: DSG, room_labels: dict) -> str:
        """
        Convert a graph into a query for the OpenAI API
        """
        room_nodes = list(scene_graph.get_layer(Layers.ROOMS).nodes)
        room_edges = list(scene_graph.get_layer(Layers.ROOMS).edges)

        object_nodes = list(scene_graph.get_layer(Layers.OBJECTS).nodes)
        object_edges = [
            edge
            for edge in scene_graph.interlayer_edges
            if NodeSymbol(edge.target).category == "O"
        ]

        objects_per_room: dict[int, list[tuple[int, str]]] = {
            room.id.category_id: [] for room in room_nodes
        }

        room_str_list: list[str] = []
        room_edges_str_list: list[str] = []

        for node in room_nodes:
            label = " ".join(room_labels[node.attributes.name].split("_"))
            room_str_list.append(f"{node.id.category_id}: {label}")
            connected_nodes = [
                NodeSymbol(edge.target).category_id
                for edge in room_edges
                if node.id.value == edge.source
            ] + [
                NodeSymbol(edge.source).category_id
                for edge in room_edges
                if node.id.value == edge.target
            ]
            room_edges_str_list.append(f"{node.id.category_id}: {connected_nodes}")

        for edge in object_edges:
            objects_per_room[NodeSymbol(edge.source).category_id].append(
                (
                    NodeSymbol(edge.target).category_id,
                    scene_graph.get_node(edge.target).attributes.name,
                )
            )

        objects_per_room_str = [f"{i[0]}: {i[1]}" for i in objects_per_room.items()]
        with open(self.environment_path) as f:
            text = f.read()
        text = text.replace("</room_str>", "; ".join(room_str_list))
        text = text.replace("</room_edges_str>", "; ".join(room_edges_str_list))
        text = text.replace("</objects_per_room_str>", "\n".join(objects_per_room_str))

        return text
