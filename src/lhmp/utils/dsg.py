from enum import IntEnum, unique
import networkx as nx
import numpy as np
from lhmp.utils.math import euclidean_distance

import spark_dsg as dsg
import spark_dsg.networkx as dsg_nx
from spark_dsg._dsg_bindings import NodeSymbol, DynamicSceneGraph as DSG
from scipy.spatial import cKDTree


@unique
class Layers(IntEnum):
    MESH = 1
    OBJECTS = 2
    PLACES = 3
    ROOMS = 4
    BUILDINGS = 5


def get_closest_node(
    places_layer_nx: nx.Graph, position: list[float], kdtree: cKDTree = None
):
    if kdtree is not None:
        closest_idx = kdtree.query(position)[1]
        closest_node = list(places_layer_nx.nodes(data=True))[closest_idx][0]
        distance = euclidean_distance(
            position, places_layer_nx.nodes[closest_node]["position"][: len(position)]
        )
    else:
        closest_node = None
        distance = np.inf
        for node, data in places_layer_nx.nodes(data=True):
            node_distance = euclidean_distance(
                position, data["position"][: len(position)]
            )
            if node_distance < distance:
                closest_node = node
                distance = node_distance
    return closest_node, distance


def construct_places_layer_nx(scene_graph: DSG) -> None:
    places_layer_nx = dsg_nx.layer_to_networkx(scene_graph.get_layer(Layers.PLACES))
    for node in places_layer_nx.nodes:
        room_id = [
            NodeSymbol(i.source).category_id
            for i in scene_graph.interlayer_edges
            if i.target == node
        ]
        if room_id:
            room_id = room_id[0]
        else:
            room_id = -1

        places_layer_nx.nodes[node]["room_id"] = room_id

    return places_layer_nx
