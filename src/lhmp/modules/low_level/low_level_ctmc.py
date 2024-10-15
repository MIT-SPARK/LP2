import spark_dsg as dsg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import random, uniform
import logging
from collections import Counter

from sklearn.neighbors import KernelDensity
from scipy.interpolate import splprep, splev, interp1d
import networkx as nx

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm

from spark_dsg._dsg_bindings import NodeSymbol, DynamicSceneGraph as DSG

from lhmp.modules.low_level.low_level_module_base import LowLevelModuleBase
from lhmp.utils.general import read_json, performance_measure
from lhmp.utils.data import GoalSequences, CoarsePaths, SmoothPath, SmoothPaths, Tree


class LowLevelCTMC(LowLevelModuleBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "walking_speed" in kwargs:
            self.walking_speed = kwargs["walking_speed"]
        else:
            self.walking_speed = 1.5  # [m/s], average human walking speed 1.5 m/s
        if "interaction_time" in kwargs:
            self.interaction_time = kwargs["interaction_time"]
        else:
            self.interaction_time = (
                5.0  # [s], current heuristic: 5 seconds per interaction
            )
        if "bandwidth" in kwargs:
            self.bandwidth = kwargs["bandwidth"]
        else:
            self.bandwidth = 3.0

        if "deterministic" in kwargs:
            self.deterministic = kwargs["deterministic"]
        else:
            self.deterministic = False

        self.step = 0.1

        if "use_diverse_sampling" in kwargs:
            self.use_diverse_sampling = kwargs["use_diverse_sampling"]
        else:
            self.use_diverse_sampling = False

        if "diverse_sampling_factor" in kwargs:
            self.diverse_sampling_factor = kwargs["diverse_sampling_factor"]
        else:
            self.diverse_sampling_factor = 0.8

    def process(
        self,
        scene_graph: DSG,
        past_trajectory: pd.DataFrame,
        coarse_paths: CoarsePaths,
        time_horizon: float,
        goal_distribution: GoalSequences,
        eval_timesteps: np.ndarray,
        *args,
        **kwargs,
    ) -> tuple[SmoothPaths, float]:
        """
        Find distribution over future trajectories
        """
        transition_tree = coarse_paths.transition_tree
        if "precomp_kdes" in kwargs and kwargs["precomp_kdes"] is not None:
            kdes = kwargs["precomp_kdes"]
        else:
            kdes = self.ctmc(
                transition_tree, past_trajectory.iloc[-1]["t"], eval_timesteps
            )

        interaction_time_0 = goal_distribution.tree.data.nodes[0]["duration"]
        if interaction_time_0 <= 0.1:
            interaction_time_0 = 0.1
            goal_distribution.tree.data.nodes[0]["duration"] = 0.1
        interaction_times = [
            [i.duration for i in goal_sequence] for goal_sequence in goal_distribution
        ]
        smooth_paths, prediction_time_horizon = self.interpolate_paths(
            scene_graph,
            past_trajectory,
            coarse_paths,
            interaction_time_0,
            interaction_times,
            time_horizon,
        )

        smooth_paths.kdes = kdes

        if "compute_steady_state" in kwargs and kwargs["compute_steady_state"]:
            Q_matrix, initial_state, node_to_state, state_to_node = self.build_q_matrix(
                transition_tree
            )
            steady_state = np.linalg.lstsq(
                np.vstack((Q_matrix, np.ones(len(Q_matrix)).reshape(1, -1))),
                np.vstack(
                    (
                        np.zeros(len(Q_matrix)).reshape(-1, 1),
                        np.ones([1]).reshape(-1, 1),
                    )
                ),
            )[0].squeeze()
            steady_state = np.abs(steady_state)
            steady_state = steady_state / steady_state.sum()
            smooth_paths.steady_state = steady_state

        return smooth_paths, prediction_time_horizon

    def ctmc(self, transition_tree: Tree, t0, eval_timesteps: np.ndarray):
        Q_matrix, initial_state, node_to_state, state_to_node = self.build_q_matrix(
            transition_tree
        )

        reachable_nodes = set([transition_tree.root])

        kdes = []

        # # calculate steady state distribution of CTMC:
        # p_steady = np.linalg.lstsq(np.vstack((Q_matrix, np.ones(len(Q_matrix)).reshape(1,-1))), np.vstack((np.zeros(len(Q_matrix)).reshape(-1,1), np.ones([1]).reshape(-1,1))))[0].squeeze()

        Q_matrix = csc_matrix(Q_matrix)

        for idx, t in enumerate(eval_timesteps):
            time = t - t0
            P_matrix = expm(Q_matrix * time)
            state_dist = P_matrix.dot(initial_state)

            reachachable_node_indices = np.where(state_dist > 1e-14)[0]
            reachable_nodes = [state_to_node[i] for i in reachachable_node_indices]

            samples = np.array(
                [
                    transition_tree.data.nodes[node]["position"][:2]
                    for node in reachable_nodes
                ]
            )
            sample_weights = state_dist[reachachable_node_indices]
            sample_weights = sample_weights / sample_weights.sum()

            # kde = KernelDensity(kernel='gaussian', bandwidth="scott")
            # kde.fit(samples, sample_weight=sample_weights)
            # bw = kde.bandwidth_

            kdes.append(
                {
                    "t": t,
                    "state_dist": state_dist,
                    "state_to_node": state_to_node,
                    "samples": samples,
                    "sample_weights": sample_weights,
                }
            )

        return kdes

    def build_q_matrix(self, transition_tree: Tree):

        num_states = len(transition_tree.data.nodes)
        Q_matrix = np.zeros((num_states, num_states))

        node_to_state = {
            node: idx for idx, node in enumerate(transition_tree.data.nodes)
        }
        state_to_node = {
            idx: node for idx, node in enumerate(transition_tree.data.nodes)
        }

        initial_state = np.zeros(num_states)
        initial_state[node_to_state[transition_tree.root]] = 1.0

        # Populate q-matrix
        for node in transition_tree.data.nodes:
            state_idx = node_to_state[node]
            rate_sum = 0

            for neighbor in transition_tree.data.neighbors(node):
                neighbor_idx = node_to_state[neighbor]
                transition_rate = transition_tree.data[node][neighbor][
                    "transition_rate"
                ]
                Q_matrix[neighbor_idx, state_idx] = transition_rate
                rate_sum += transition_rate

            Q_matrix[state_idx, state_idx] = -rate_sum

        return Q_matrix, initial_state, node_to_state, state_to_node

    def interpolate_paths(
        self,
        scene_graph: DSG,
        past_trajectory: pd.DataFrame,
        coarse_paths: CoarsePaths,
        interaction_time_0: float,
        interaction_times: list[list[float]],
        time_horizon: float,
    ) -> tuple[SmoothPaths, float]:

        print("interaction_time_0: ", interaction_time_0)
        logging.info(f"CTMC interaction time zero: {interaction_time_0}")

        # number of discrete paths up until first interaction:
        # paths_proba = sorted(coarse_paths.paths, key = lambda x: x.probability)
        # count_20 = Counter([tuple(p.path[0]) for p in paths_proba][:20])
        # print(f"Number of discrete paths up until first interaction in top 20 predictions: {count_20}")

        diverse_sampling_lookups = []

        min_horizon = np.inf
        paths_interpolated = SmoothPaths()
        for path_idx, path in enumerate(coarse_paths):

            if self.use_diverse_sampling:
                first_seg = tuple(path.path[0])
                if first_seg in diverse_sampling_lookups:
                    path.probability = path.probability * self.diverse_sampling_factor
                diverse_sampling_lookups.append(first_seg)

            time = past_trajectory.iloc[-1]["t"]
            path_interpolated = {
                "x": [past_trajectory.iloc[-1]["x"]],
                "y": [past_trajectory.iloc[-1]["y"]],
                "t": [time],
                "room_id": [past_trajectory.iloc[-1]["room_id"]],
                "probability": [path.probability],
            }

            interaction_time = np.clip(interaction_time_0, 0.1, 60.0)

            logging.info(
                f"sampled interaction time '{interaction_time}' out of exponential distribution with mean '{interaction_time_0}'"
            )
            times = np.arange(time, time + interaction_time, self.step)
            path_interpolated["x"].extend(
                np.ones_like(times) * past_trajectory.iloc[-1]["x"]
            )
            path_interpolated["y"].extend(
                np.ones_like(times) * past_trajectory.iloc[-1]["y"]
            )
            path_interpolated["t"].extend(times)
            path_interpolated["room_id"].extend(
                np.ones_like(times) * past_trajectory.iloc[-1]["room_id"]
            )
            path_interpolated["probability"].extend(
                np.ones_like(times) * path.probability
            )

            time = times[-1]

            for seg_idx, path_segment in enumerate(path):
                if len(path.path) == 1 and len(path_segment) == 1:
                    continue
                time += self.step
                if seg_idx == 0:
                    coarse_data = {
                        "x": [past_trajectory.iloc[-1]["x"]],
                        "y": [past_trajectory.iloc[-1]["y"]],
                        "room_id": [past_trajectory.iloc[-1]["room_id"]],
                    }
                else:
                    coarse_data = {"x": [], "y": [], "room_id": []}
                for node_idx, node in enumerate(path_segment):
                    x_pos, y_pos, room_id = self.get_node_data(scene_graph, node)
                    coarse_data["x"].append(
                        x_pos
                    )  # avoids degenerate configurations for KDE
                    coarse_data["y"].append(y_pos)
                    coarse_data["room_id"].append(room_id)

                if len(path_segment) < 2:
                    if seg_idx == 0:
                        x_pos, y_pos, room_id = self.get_node_data(
                            scene_graph, path.path[seg_idx + 1][0]
                        )
                        coarse_data["x"].append(x_pos)
                        coarse_data["y"].append(y_pos)
                        coarse_data["room_id"].append(room_id)
                    else:
                        x_pos, y_pos, room_id = self.get_node_data(
                            scene_graph, path.path[seg_idx - 1][-1]
                        )
                        coarse_data["x"].insert(0, x_pos)
                        coarse_data["y"].insert(0, y_pos)
                        coarse_data["room_id"].insert(0, room_id)

                new_positions, new_room_ids = self.interpolate_path(
                    np.c_[coarse_data["x"], coarse_data["y"]],
                    np.array(coarse_data["room_id"]),
                )
                new_times = np.linspace(
                    time,
                    time + len(new_positions) * self.step,
                    len(new_positions),
                    endpoint=False,
                )
                new_probabilities = np.ones(len(new_positions)) * path.probability

                time = new_times[-1]
                path_interpolated["x"].extend(new_positions[:, 0])
                path_interpolated["y"].extend(new_positions[:, 1])
                path_interpolated["t"].extend(new_times)
                path_interpolated["room_id"].extend(new_room_ids)
                path_interpolated["probability"].extend(new_probabilities)

                node = path_segment[-1]
                x_pos, y_pos, room_id = self.get_node_data(scene_graph, node)

                interaction_time = np.clip(
                    interaction_times[path_idx][seg_idx], 0.1, 60.0
                )  # np.random.exponential(scale=np.clip(interaction_times[path_idx][seg_idx], 0.1, 60.0))

                logging.info(
                    f"sampled interaction time '{interaction_time}' out of exponential distribution with mean '{interaction_times[path_idx][seg_idx]}'"
                )
                for _ in range(int(interaction_time / self.step)):
                    time += self.step
                    path_interpolated["x"].append(x_pos)
                    path_interpolated["y"].append(y_pos)
                    path_interpolated["t"].append(time)
                    path_interpolated["room_id"].append(room_id)
                    path_interpolated["probability"].append(path.probability)

            min_horizon = min(min_horizon, time)

            time += self.step
            if time < time_horizon:
                extra_time = [
                    round(i, 4)
                    for i in np.arange(time, time_horizon, self.step).tolist()
                ]
                path_interpolated["t"].extend(extra_time)
                path_interpolated["x"].extend(
                    [path_interpolated["x"][-1]] * len(extra_time)
                )
                path_interpolated["y"].extend(
                    [path_interpolated["y"][-1]] * len(extra_time)
                )
                path_interpolated["room_id"].extend(
                    [path_interpolated["room_id"][-1]] * len(extra_time)
                )
                path_interpolated["probability"].extend(
                    [path_interpolated["probability"][-1]] * len(extra_time)
                )

            paths_interpolated.paths.append(
                SmoothPath(
                    path=pd.DataFrame(path_interpolated), probability=path.probability
                )
            )
            min_horizon = min(min_horizon, time)

        paths_interpolated.predicted_time_horizon = min_horizon

        return paths_interpolated, min_horizon

    def get_node_data(self, scene_graph: DSG, node: int):
        node_attributes = scene_graph.get_node(node).attributes
        room_edge = [
            edge for edge in scene_graph.interlayer_edges if edge.target == node
        ]
        if room_edge:
            room_id = NodeSymbol(room_edge[0].source).category_id
        else:
            room_id = 0
        x_pos = node_attributes.position[0]
        y_pos = node_attributes.position[1]

        return x_pos, y_pos, room_id

    def compute_time(self, distance: float):
        return distance / self.walking_speed

    def interpolate_path(
        self, path: np.ndarray[np.float64], room_ids: np.ndarray
    ) -> tuple[np.ndarray[np.float64], np.ndarray[np.int32]]:
        """
        interpolate path positions into smooth trajectory,
        sample at rate self.step assuming constant velocity of self.human_velocity
        """
        try:
            if len(path) < 5:
                raise Exception("Trajectory too short")
            tck_pos, u_pos = splprep(path.T, s=0)
            path_length = self.path_length_spline(tck_pos, u_pos)
            u_pos_new = np.linspace(
                u_pos.min(),
                u_pos.max(),
                int(path_length / (self.walking_speed * self.step)),
            )
            new_positions = np.array(splev(u_pos_new, tck_pos)).T
            new_room_ids = room_ids[
                np.floor(u_pos_new * (len(room_ids) - 1)).astype(int)
            ]
        except Exception as e:
            print(
                "Trajectory could not be interpolated using splines, falling back to linear interpolation"
            )
            total_points = max(
                len(path),
                int(
                    sum(
                        [
                            np.linalg.norm(path[i] - path[i + 1])
                            for i in range(len(path) - 1)
                        ]
                    )
                    / (self.walking_speed * self.step)
                ),
            )
            indices = np.linspace(0, 1, len(path))
            indices_new = np.linspace(0, 1, total_points)

            x_coords = interp1d(indices, path[:, 0], kind="linear")(indices_new)
            y_coords = interp1d(indices, path[:, 1], kind="linear")(indices_new)
            new_positions = np.c_[x_coords, y_coords]

            new_room_ids = interp1d(indices, room_ids, kind="nearest")(indices_new)
            new_room_ids = new_room_ids.astype(int)

        return new_positions, new_room_ids

    def path_length_spline(self, tck, u):
        """
        Calculate the length of a path given its spline representation
        """
        return np.sum(
            np.sqrt(np.sum(np.square(np.diff(splev(u, tck), axis=1)), axis=0))
        )
