import numpy as np
import pandas as pd
import os
import json
from typing import Optional, Union

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import matplotlib.colors as colors
import plotly.graph_objects as go
import matplotlib.patheffects as path_effects
from matplotlib.animation import FuncAnimation
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap

plt.rcParams.update({"font.size": 15})

from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

import spark_dsg as dsg
from spark_dsg._dsg_bindings import DynamicSceneGraph as DSG
import networkx as nx

from lhmp.utils.dsg import Layers
from lhmp.utils.data import reindex_trajectory, SmoothPath
from lhmp.utils.general import read_json, unity_coordinates_to_right_hand_coordinates
from lhmp.utils.data import (
    SmoothPath,
    SmoothPaths,
    GoalSequences,
    CoarsePaths,
    CoarsePath,
    Tree,
)


LAYER_SETTINGS = {
    Layers.BUILDINGS.value: {"marker_size": 8},
    Layers.ROOMS.value: {"marker_size": 6},
    Layers.OBJECTS.value: {"marker_size": 4},
    Layers.PLACES.value: {"marker_size": 4},
}

NODE_TYPE_TO_COLOR = {"B": "#636EFA", "R": "#EF553B", "p": "#AB63FA", "O": "#00CC96"}


def generate_colors(num_colors: int, normalization=False):
    """
    Generate a list of 3D numpy arrays of RGB colors maximizing distance on hue wheel.

    Parameters:
    - num_colors: int
      The number of colors to generate.

    Returns:
    - A list of 3D numpy arrays of RGB colors.
    """
    if num_colors <= 0:
        return []

    hues = np.linspace(0, 1, num_colors, endpoint=False)
    saturation = 1.0
    value = 1.0
    hsv_colors = np.column_stack(
        (hues, np.full_like(hues, saturation), np.full_like(hues, value))
    )
    rgb_colors = [hsv_to_rgb(hsv) for hsv in hsv_colors]
    if not normalization:
        rgb_colors = [np.round(rgb * 255) for rgb in rgb_colors]
    rgb_colors = [tuple(np.reshape(rgb, (3,))) for rgb in rgb_colors]

    return tuple(rgb_colors)


def visualize_path(
    fig: go.Figure, scene_graph: DSG, path: list[int], offset: bool = True
) -> go.Figure:

    x_lines, y_lines, z_lines = [], [], []

    for idx in range(len(path) - 1):
        edge_source = path[idx]
        edge_target = path[idx + 1]
        source = scene_graph.get_node(edge_source)
        target = scene_graph.get_node(edge_target)

        if offset:
            start_pos = dsg.visualization.z_offset(source)
            end_pos = dsg.visualization.z_offset(target)
        else:
            start_pos = source.attributes.position.copy()
            end_pos = target.attributes.position.copy()

        x_lines += [start_pos[0], end_pos[0], None]
        y_lines += [start_pos[1], end_pos[1], None]
        z_lines += [start_pos[2], end_pos[2], None]

    fig.add_trace(
        go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode="lines",
            line=dict(color="rgb(255,0,0)", width=12),
            hoverinfo="none",
        )
    )
    return fig


def visualize_dsg_layer(
    fig: go.Figure,
    scene_graph_layer: dsg._dsg_bindings.LayerView,
    marker_size=12,
    annotation_size=18,
    include_text=False,
    text_func=None,
    color_func=None,
) -> go.Figure:

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(0,0,0,0)",
                showgrid=False,
                zerolinecolor="rgba(0,0,0,0)",
            ),
            yaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(0,0,0,0)",
                showgrid=False,
                zerolinecolor="rgba(0,0,0,0)",
            ),
            zaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(0,0,0,0)",
                showgrid=False,
                zerolinecolor="rgba(0,0,0,0)",
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )

    pos, colors, text = [], [], []
    x_lines_dark, y_lines_dark, z_lines_dark = [], [], []

    for node in scene_graph_layer.nodes:
        pos.append(np.squeeze(node.attributes.position.copy()))
        if color_func is None:
            colors.append(NODE_TYPE_TO_COLOR[node.id.category])
        else:
            colors.append(color_func(node))

        if text_func is None:
            text.append(str(node.id))
        else:
            text.append(text_func(node))

    pos = np.array(pos)
    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            text=text if include_text else "",
            mode="markers",
            marker=dict(color=colors, size=marker_size, opacity=0.8),
            hoverinfo="none",
        )
    )

    for edge in scene_graph_layer.edges:
        source = scene_graph_layer.get_node(edge.source)
        target = scene_graph_layer.get_node(edge.target)

        start_pos = source.attributes.position.copy()
        end_pos = target.attributes.position.copy()

        x_lines_dark += [start_pos[0], end_pos[0], None]
        y_lines_dark += [start_pos[1], end_pos[1], None]
        z_lines_dark += [start_pos[2], end_pos[2], None]

    fig.add_trace(
        go.Scatter3d(
            x=x_lines_dark,
            y=y_lines_dark,
            z=z_lines_dark,
            mode="lines",
            line=dict(color="rgb(0,0,0)", width=1),
            hoverinfo="none",
        )
    )

    return fig


def plot_trajectory(
    fig, data: pd.DataFrame, color="rgb(0,255,0)", offset=10
) -> go.Figure:
    pos = []
    for index, row in data.iterrows():
        pos.append(np.squeeze(np.array([row["x"], row["y"], offset])))

    pos = np.array(pos)
    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="markers",
            marker=dict(color=color, size=3),
            hoverinfo="none",
        )
    )
    return fig


def visualize_paths_in_dsg(
    fig,
    paths: list[pd.DataFrame],
    past_trajectory: pd.DataFrame,
    gt_future_trajectory: pd.DataFrame,
    offset=10,
):
    # predictions:
    for path in paths:
        fig = plot_trajectory(fig, path, color="rgb(155,0,0)", offset=offset)
    # gt:
    fig = plot_trajectory(fig, past_trajectory, color="rgb(255,190,0)", offset=offset)
    fig = plot_trajectory(fig, gt_future_trajectory, offset=offset)

    return fig


def visualize_interaction_prediction_tree(
    interaction_sequences_semantics_tree: nx.DiGraph, key="semantic_label"
):
    black_edges = [edge for edge in interaction_sequences_semantics_tree.edges()]
    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    pos = nx.multipartite_layout(
        interaction_sequences_semantics_tree, subset_key="depth"
    )
    nx.draw_networkx_nodes(
        interaction_sequences_semantics_tree,
        pos,
        cmap=plt.get_cmap("jet"),
        node_size=500,
    )
    nx.draw_networkx_labels(
        interaction_sequences_semantics_tree,
        pos,
        labels={
            node: atr[key] + f" {round(atr['duration'],1)}"
            for node, atr in interaction_sequences_semantics_tree.nodes.items()
            if node != 0
        },
    )
    nx.draw_networkx_edges(
        interaction_sequences_semantics_tree, pos, edgelist=black_edges, arrows=False
    )
    plt.show()


def draw_places_layer(ax: plt.Axes, places_layer: nx.DiGraph) -> plt.Axes:
    pos, colors, text = [], [], []
    x_lines_dark, y_lines_dark, z_lines_dark = [], [], []

    for node in places_layer.nodes:
        pos.append(np.squeeze(node.attributes.position.copy()))

    for edge in places_layer.edges:
        source = places_layer.get_node(edge.source)
        target = places_layer.get_node(edge.target)

        start_pos = source.attributes.position.copy()
        end_pos = target.attributes.position.copy()

        x_lines_dark += [start_pos[0], end_pos[0], None]
        y_lines_dark += [start_pos[1], end_pos[1], None]

    pos = np.array(pos)

    ax.plot(
        x_lines_dark, y_lines_dark, c="lightgrey", marker="", linewidth=0.25, alpha=0.5
    )
    ax.scatter(pos[:, 0], pos[:, 1], c="w", s=24.0)
    return ax


def animation_vis(
    places_layer: dsg._dsg_bindings.LayerView,
    predicted_trajectories: list[SmoothPath],
    gt_future_trajectory: pd.DataFrame,
    env_AABB: list[tuple[float, float]],
    t_init: float,
    t_stop: float,
    bg_cfg: Optional[dict] = None,
) -> None:

    if bg_cfg is not None:
        img = plt.imread(bg_cfg["floorplan_image"])
        fig, ax = plt.subplots(figsize=(8, img.shape[0] / img.shape[1] * 8))

        bottom_left_3D = unity_coordinates_to_right_hand_coordinates(
            bg_cfg["geometry_unity"]["bottom_left"][0],
            bg_cfg["geometry_unity"]["bottom_left"][1],
            bg_cfg["geometry_unity"]["bottom_left"][2],
        )
        top_right_3D = unity_coordinates_to_right_hand_coordinates(
            bg_cfg["geometry_unity"]["top_right"][0],
            bg_cfg["geometry_unity"]["top_right"][1],
            bg_cfg["geometry_unity"]["top_right"][2],
        )

        bottom_left_2D = bg_cfg["image_coordinates"]["bottom_left"]
        top_right_2D = bg_cfg["image_coordinates"]["top_right"]

        x = np.linspace(0, img.shape[1], 100)
        y = np.linspace(0, img.shape[0], 100)

        x_3d, y_3d = projection_2d_to_3d(
            x, y, bottom_left_3D, top_right_3D, bottom_left_2D, top_right_2D, img.shape
        )

        X, Y = np.meshgrid(x_3d, y_3d)

        grid = np.c_[X.ravel(), Y.ravel()]
        X_2d, Y_2d = np.meshgrid(x, y)
        ax.imshow(img, alpha=0.7, origin="lower")

        plt.xlim([0, img.shape[1]])
        plt.ylim([0, img.shape[0]])

        ax.set_xticks([])
        ax.set_yticks([])

    else:
        x = np.linspace(env_AABB[0][0], env_AABB[1][0], 100)
        y = np.linspace(env_AABB[0][1], env_AABB[1][1], 100)
        X, Y = np.meshgrid(x, y)

        figsize = (
            10,
            (env_AABB[1][1] - env_AABB[0][1]) / (env_AABB[1][0] - env_AABB[0][0]) * 10,
        )
        fig, ax = plt.subplots(figsize=figsize)
        ax = draw_places_layer(ax, places_layer)

        plt.xlim([env_AABB[0][0], env_AABB[1][0]])
        plt.ylim([env_AABB[0][1], env_AABB[1][1]])

        grid = np.c_[X.ravel(), Y.ravel()]

    new_timesteps = np.arange(
        gt_future_trajectory["t"].iloc[0] + t_init,
        min(
            gt_future_trajectory["t"].iloc[-1],
            gt_future_trajectory["t"].iloc[0] + t_stop,
        ),
        0.1,
    )
    gt_trajectory_reindexed = reindex_trajectory(
        gt_future_trajectory,
        new_timesteps,
        linear_keys=["x", "y", "z"],
        nearest_keys=["interaction_id", "room_id"],
    )
    paths_reindexed = [
        reindex_trajectory(
            path.path, new_timesteps, linear_keys=["x", "y"], nearest_keys=["room_id"]
        )
        for path in predicted_trajectories
    ]

    # first step
    sample0 = np.array(
        [
            path_reindexed[path_reindexed["t"] == new_timesteps[0]][["x", "y"]]
            .to_numpy()
            .squeeze()
            for path_reindexed in paths_reindexed
        ]
    )

    t0 = new_timesteps[0]

    kde = KernelDensity(kernel="gaussian", bandwidth="scott")
    kde.fit(sample0)

    bw0 = kde.bandwidth_

    if bg_cfg is not None:
        Z = np.exp(kde.score_samples(grid)).reshape(X_2d.shape)

        cmap_dist = create_custom_cmap(pl.cm.OrRd)
        cmap_traj = create_custom_cmap(pl.cm.Greens)

        pcm = ax.pcolor(
            X_2d,
            Y_2d,
            Z,
            norm=colors.LogNorm(vmin=1e-6, vmax=10),
            cmap=cmap_dist,
            shading="nearest",
            edgecolors="face",
        )
    else:
        Z = np.exp(kde.score_samples(grid)).reshape(X.shape)
        pcm = ax.pcolor(
            X,
            Y,
            Z,
            norm=colors.LogNorm(vmin=1e-6, vmax=10),
            cmap="inferno",
            edgecolors="face",
        )
    Z = Z.clip(min=1e-6)

    plt.colorbar(pcm, ax=ax, label="Density")
    ax.set_title(f"Probability Distribution for t = {t_init}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    t_min = new_timesteps[0]
    t_max = new_timesteps[-1]
    norm = plt.Normalize(t_min, t_max, clip=True)

    def update(t: float):
        x_traj, y_traj = projection_3d_to_2d(
            gt_trajectory_reindexed[gt_trajectory_reindexed["t"] <= t]["x"].to_numpy()[
                -100:
            ],
            gt_trajectory_reindexed[gt_trajectory_reindexed["t"] <= t]["y"].to_numpy()[
                -100:
            ],
            bottom_left_3D,
            top_right_3D,
            bottom_left_2D,
            top_right_2D,
            img.shape,
        )
        sample = np.array(
            [
                path_reindexed[path_reindexed["t"] == t][["x", "y"]]
                .to_numpy()
                .squeeze()
                for path_reindexed in paths_reindexed
            ]
        )

        points_2d_x, points_2d_y = projection_3d_to_2d(
            sample[:, 0],
            sample[:, 1],
            bottom_left_3D,
            top_right_3D,
            bottom_left_2D,
            top_right_2D,
            img.shape,
        )

        timeval = (
            gt_trajectory_reindexed[gt_trajectory_reindexed["t"] <= t]["t"]
        ).to_numpy()

        plt.scatter(
            x_traj,
            y_traj,
            c=timeval[-100:],
            cmap=pl.cm.BuGn,
            norm=norm,
            marker=".",
            s=24.0,
        )
        plt.scatter(
            points_2d_x,
            points_2d_y,
            c=np.ones_like(points_2d_x) * t,
            cmap=pl.cm.YlOrRd,
            norm=norm,
            marker=".",
            s=24.0,
        )

        kde = KernelDensity(kernel="gaussian", bandwidth=bw0)
        kde.fit(sample)

        if bg_cfg is not None:
            Z = np.exp(kde.score_samples(grid)).reshape(X_2d.shape)
        else:
            Z = np.exp(kde.score_samples(grid)).reshape(X.shape)
        Z = Z.clip(min=1e-6)
        pcm.set_array(Z.ravel())
        ax.set_title(f"Probability Distribution for t = {round(t-t0,4)}")
        fig.canvas.draw_idle()
        return pcm

    ani = FuncAnimation(fig, update, frames=new_timesteps[1:590][::5], repeat=False)
    return ani


def animation_vis_unstructured(
    places_layer: dsg._dsg_bindings.LayerView,
    kde_data: list[dict],
    discrete_data: list[SmoothPath],
    gt_trajectory_reindexed: pd.DataFrame,
    env_AABB: list[tuple[float, float]],
    t_init: float,
    t_stop: float,
    bg_cfg: Optional[dict] = None,
    bandwidth: float = 0.6,
) -> None:

    if bg_cfg is not None:
        img = plt.imread(bg_cfg["floorplan_image"])
        fig, ax = plt.subplots(figsize=(8, img.shape[0] / img.shape[1] * 8))

        bottom_left_3D = unity_coordinates_to_right_hand_coordinates(
            bg_cfg["geometry_unity"]["bottom_left"][0],
            bg_cfg["geometry_unity"]["bottom_left"][1],
            bg_cfg["geometry_unity"]["bottom_left"][2],
        )
        top_right_3D = unity_coordinates_to_right_hand_coordinates(
            bg_cfg["geometry_unity"]["top_right"][0],
            bg_cfg["geometry_unity"]["top_right"][1],
            bg_cfg["geometry_unity"]["top_right"][2],
        )

        bottom_left_2D = bg_cfg["image_coordinates"]["bottom_left"]
        top_right_2D = bg_cfg["image_coordinates"]["top_right"]

        x = np.linspace(0, img.shape[1], 100)
        y = np.linspace(0, img.shape[0], 100)

        x_3d, y_3d = projection_2d_to_3d(
            x, y, bottom_left_3D, top_right_3D, bottom_left_2D, top_right_2D, img.shape
        )

        X, Y = np.meshgrid(x_3d, y_3d)

        grid = np.c_[X.ravel(), Y.ravel()]
        X_2d, Y_2d = np.meshgrid(x, y)
        ax.imshow(img, alpha=0.7, origin="lower")

        plt.xlim([0, img.shape[1]])
        plt.ylim([0, img.shape[0]])

        ax.set_xticks([])
        ax.set_yticks([])
    else:
        raise ValueError(
            "Background configuration is required for unstructured environment"
        )

    new_timesteps = np.array([entry["t"] for entry in kde_data])
    new_timesteps_discrete = np.arange(new_timesteps[0], new_timesteps[-1], 0.4)
    paths_reindexed = [
        reindex_trajectory(
            path.path,
            new_timesteps_discrete,
            linear_keys=["x", "y"],
            nearest_keys=["room_id"],
        )
        for path in discrete_data
    ]

    # first step
    t0 = new_timesteps[0]

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(kde_data[0]["samples"], sample_weight=kde_data[0]["sample_weights"])

    Z = np.exp(kde.score_samples(grid)).reshape(X_2d.shape)
    Z = Z.clip(min=1e-6)

    cmap_dist = create_custom_cmap(pl.cm.OrRd)

    t_min = new_timesteps[0]
    t_max = new_timesteps[-1]
    norm = plt.Normalize(t_min - 20.0, t_min + 60.0, clip=True)

    pcm = ax.pcolor(
        X_2d,
        Y_2d,
        Z,
        norm=colors.LogNorm(vmin=1e-6, vmax=10),
        cmap=cmap_dist,
        shading="nearest",
        edgecolors="face",
    )
    plt.colorbar(pcm, ax=ax, label="Density")
    ax.set_title(f"Probability Distribution for t = {t_init}")

    def update(entry: list[dict]):
        x_traj, y_traj = projection_3d_to_2d(
            gt_trajectory_reindexed[gt_trajectory_reindexed["t"] <= entry["t"]][
                "x"
            ].to_numpy()[-100:],
            gt_trajectory_reindexed[gt_trajectory_reindexed["t"] <= entry["t"]][
                "y"
            ].to_numpy()[-100:],
            bottom_left_3D,
            top_right_3D,
            bottom_left_2D,
            top_right_2D,
            img.shape,
        )

        timeval = (
            gt_trajectory_reindexed[gt_trajectory_reindexed["t"] <= entry["t"]]["t"]
        ).to_numpy()

        plt.scatter(
            x_traj,
            y_traj,
            c=timeval[-100:],
            cmap=pl.cm.BuGn,
            norm=norm,
            marker=".",
            s=24.0,
        )

        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(entry["samples"], sample_weight=entry["sample_weights"])
        Z = np.exp(kde.score_samples(grid)).reshape(X_2d.shape)
        Z = Z.clip(min=1e-6)
        pcm.set_array(Z.ravel())
        ax.set_title(f"Probability Distribution for t = {round(entry['t']-t0,4)}")
        fig.canvas.draw_idle()
        return pcm

    ani = FuncAnimation(fig, update, frames=kde_data, repeat=False)
    return ani


def animation_vis_discrete(
    data: list[SmoothPath],
    past_trajectory: pd.DataFrame,
    gt_trajectory_reindexed: pd.DataFrame,
    t_init: float,
    t_stop: float,
    bg_cfg: dict,
) -> None:

    img = plt.imread(bg_cfg["floorplan_image"])
    fig, ax = plt.subplots(figsize=(8, img.shape[0] / img.shape[1] * 8))

    bottom_left_3D = unity_coordinates_to_right_hand_coordinates(
        bg_cfg["geometry_unity"]["bottom_left"][0],
        bg_cfg["geometry_unity"]["bottom_left"][1],
        bg_cfg["geometry_unity"]["bottom_left"][2],
    )
    top_right_3D = unity_coordinates_to_right_hand_coordinates(
        bg_cfg["geometry_unity"]["top_right"][0],
        bg_cfg["geometry_unity"]["top_right"][1],
        bg_cfg["geometry_unity"]["top_right"][2],
    )

    bottom_left_2D = bg_cfg["image_coordinates"]["bottom_left"]
    top_right_2D = bg_cfg["image_coordinates"]["top_right"]

    x = np.linspace(0, img.shape[1], 100)
    y = np.linspace(0, img.shape[0], 100)

    x_3d, y_3d = projection_2d_to_3d(
        x, y, bottom_left_3D, top_right_3D, bottom_left_2D, top_right_2D, img.shape
    )

    X, Y = np.meshgrid(x_3d, y_3d)

    grid = np.c_[X.ravel(), Y.ravel()]
    X_2d, Y_2d = np.meshgrid(x, y)
    ax.imshow(img, alpha=0.7, origin="lower")

    plt.xlim([0, img.shape[1]])
    plt.ylim([0, img.shape[0]])

    ax.set_xticks([])
    ax.set_yticks([])

    t0 = gt_trajectory_reindexed["t"].iloc[0]
    new_timesteps = np.array(
        [
            entry
            for entry in gt_trajectory_reindexed["t"]
            if entry - t0 <= t_stop and entry - t0 >= t_init
        ]
    )[::10]

    norm = plt.Normalize(t_init - 20.0, t_init + t_stop, clip=True)

    norm_past = plt.Normalize(
        past_trajectory["t"].iloc[0], past_trajectory["t"].iloc[-1], clip=True
    )

    x_traj_p, y_traj_p = projection_3d_to_2d(
        past_trajectory["x"].to_numpy(),
        past_trajectory["y"].to_numpy(),
        bottom_left_3D,
        top_right_3D,
        bottom_left_2D,
        top_right_2D,
        img.shape,
    )

    timeval = (past_trajectory["t"]).to_numpy()

    plt.scatter(
        x_traj_p, y_traj_p, c=timeval, cmap=pl.cm.BuGn, norm=norm, marker=".", s=24.0
    )

    ax.set_title(f"Predicted discrete trajectories for t = {t_init}")

    def update(entry: float):
        x_traj, y_traj = projection_3d_to_2d(
            gt_trajectory_reindexed[gt_trajectory_reindexed["t"] <= entry][
                "x"
            ].to_numpy()[-100:],
            gt_trajectory_reindexed[gt_trajectory_reindexed["t"] <= entry][
                "y"
            ].to_numpy()[-100:],
            bottom_left_3D,
            top_right_3D,
            bottom_left_2D,
            top_right_2D,
            img.shape,
        )

        for path in data:
            timeval = (path.path[path.path["t"] <= entry]["t"] - t0).to_numpy()[-100:]

            x_vals = path.path[path.path["t"] <= entry]["x"].to_numpy()[-100:]
            y_vals = path.path[path.path["t"] <= entry]["y"].to_numpy()[-100:]
            x_sample, y_sample = projection_3d_to_2d(
                x_vals,
                y_vals,
                bottom_left_3D,
                top_right_3D,
                bottom_left_2D,
                top_right_2D,
                img.shape,
            )
            plt.scatter(
                x_sample,
                y_sample,
                c=timeval[-100:],
                cmap=pl.cm.YlOrRd,
                norm=norm,
                marker=".",
                s=24.0,
            )

        print(f"timestep {round(entry-t0,2)} out of {t_stop-t_init}")
        ax.set_title(f"Predicted discrete trajectories for t = {round(entry-t0,2)}")

    ani = FuncAnimation(fig, update, frames=new_timesteps, repeat=False)
    return ani


def invert_data_structure(
    dirs: list[str], ids_to_include: Optional[list[int]] = None
) -> dict:
    if ids_to_include is not None:
        files = sorted(
            [
                i
                for i in os.listdir(dirs[0])
                if int(i.split(".")[0].split("_")[-1]) in ids_to_include
                and np.all([j.isdigit() for j in i.split(".")[0].split("_")[-2:]])
            ]
        )
    else:
        files = sorted(
            [
                i
                for i in os.listdir(dirs[0])
                if [j.isdigit() for j in i.split(".")[0].split("_")[-2:]]
                and np.all([j.isdigit() for j in i.split(".")[0].split("_")[-2:]])
            ]
        )
    data = {}
    for file in files:
        key = tuple(
            [int(i) for i in file.split(".")[0].split("_")[-2:] if i.isnumeric()]
        )
        data[key] = {}
        for folder in dirs:
            if file.endswith(".csv"):
                data[key][folder.split("/")[-1]] = pd.read_csv(
                    os.path.join(folder, file), index_col=0
                )
            elif file.endswith(".pkl"):
                data[key][folder.split("/")[-1]] = pd.read_pickle(
                    os.path.join(folder, file)
                )

    return data


def invert_data_structure_interaction(
    dirs: list[str], ids_to_include: Optional[list[int]] = None
) -> dict:
    if ids_to_include is not None:
        files = sorted(
            [
                i
                for i in os.listdir(dirs[0])
                if int(i.split(".")[0].split("_")[-1]) in ids_to_include
            ],
            key=lambda x: int(x.split(".")[0].split("_")[-1]),
        )
    else:
        files = sorted(
            [i for i in os.listdir(dirs[0])],
            key=lambda x: int(x.split(".")[0].split("_")[-1]),
        )
    data = {}
    for file in files:
        dataset_type, id = file.split(".")[0].split("_")[-2:]
        if dataset_type == "interaction":
            data[(dataset_type, id)] = {}
            for folder in dirs:
                if file.endswith(".csv"):
                    data[(dataset_type, id)][folder.split("/")[-1]] = pd.read_csv(
                        os.path.join(folder, file), index_col=0
                    )
                elif file.endswith(".pkl"):
                    data[(dataset_type, id)][folder.split("/")[-1]] = pd.read_pickle(
                        os.path.join(folder, file)
                    )

    return data


def plot_all_methods_per_trajectory(
    data_inv, colors, key="ade", lim=60.0, title="ADE Bo20", save_path=None
):
    N = len(data_inv.keys())

    n_cols = (N + 2) // 3
    n_rows = min(3, N)

    plt.figure(figsize=(10 * n_cols, 6 * n_rows))

    plt.title(f"{key.upper()}", fontsize=16)

    for i, (sample_id, methods) in enumerate(data_inv.items(), start=1):
        ax = plt.subplot(n_rows, n_cols, i)

        t_min = min([df["t"].iloc[-1] for df in methods.values()])

        for method_name, df in methods.items():
            df = df[df["t"] <= lim]
            ax.plot(
                np.array(df["t"].fillna("bfill")),
                np.array(df[key]),
                label=method_name,
                color=colors[method_name][0],
                linestyle=colors[method_name][1],
            )

        ax.set_xlim([0, min(t_min, lim)])
        ax.set_xlabel("Time Horizon [s]")
        ax.set_ylabel(key.upper())
        ax.set_title(f"Sample ID: {sample_id[1]}, Split {sample_id[0]}")
        ax.legend()

    plt.title(title)
    plt.tight_layout(h_pad=10.0, w_pad=2.0)
    plt.subplots_adjust(top=0.945, bottom=0.055)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_mean_and_std_multiple_scenes(
    ax: plt.Axes,
    data_inv: dict,
    colors: dict,
    key: str = "ade",
    lim: float = 60.0,
    title: str = "ADE Bo20",
    plot_std: bool = True,
):
    methods = list(next(iter(data_inv.values())).keys())

    max_t = max(
        [
            df["t"].max()
            for methods_data in data_inv.values()
            for df in methods_data.values()
        ]
    )
    values = {method: [] for method in methods}
    t_values = range(1, int(lim) + 1)

    for sample_id, methods_data in data_inv.items():
        for method_name, df in methods_data.items():
            df = df[df["t"] <= lim]
            df_aligned = reindex_trajectory(
                df, np.arange(1, int(lim) + 1), linear_keys=[key], nearest_keys=[]
            )
            values[method_name].append(df_aligned[key].values)

    means = {method: np.nanmean(vals, axis=0) for method, vals in values.items()}
    stds = {method: np.nanstd(vals, axis=0) for method, vals in values.items()}

    print("Metric: ", title)
    print({k: v[-1] for k, v in means.items()})

    for method_name in methods:
        if method_name == "places_uniform_goal":
            legend_name = "Random goal"
        elif method_name == "reachable_states":
            legend_name = "Random reachable state"
        elif method_name == "ctmc":
            legend_name = "Random walk"
        elif method_name == "v4_deterministic":
            legend_name = "Deterministic method"
        elif method_name == "v4_instance":
            legend_name = "Instance Prediction"
        elif method_name == "v4":
            legend_name = "LP$^2$ (Ours)"
        elif method_name == "v4_gt_action_sequences_instance":
            legend_name = "Given ground truth interaction sequences"
        elif method_name == "v4_gt_action_sequences_semantics":
            legend_name = "Given ground truth semantic interaction sequences"
        elif method_name == "const_vel":
            legend_name = "Constant velocity"
        elif method_name == "trajectron":
            legend_name = "Trajectron++$\\star$"
        elif method_name == "ynet":
            legend_name = "YNet$\\star$"
        else:
            legend_name = method_name
        ax.plot(
            t_values,
            means[method_name],
            label=legend_name,
            color=colors[method_name][0],
            linestyle=colors[method_name][1],
            linewidth=2,
        )

        if plot_std:
            if key == "acc":
                ax.fill_between(
                    t_values,
                    (means[method_name] - stds[method_name]).clip(min=0.0),
                    (means[method_name] + stds[method_name]).clip(max=1.0),
                    color=colors[method_name][0],
                    linestyle=colors[method_name][1],
                    linewidth=2,
                    alpha=0.2,
                )
            else:
                ax.fill_between(
                    t_values,
                    means[method_name] - stds[method_name],
                    means[method_name] + stds[method_name],
                    color=colors[method_name][0],
                    linestyle=colors[method_name][1],
                    linewidth=2,
                    alpha=0.2,
                )

    return ax


def plot_mean_and_std(
    data_inv: dict,
    colors: dict,
    key: str = "ade",
    lim: float = 60.0,
    ylim: list[int] = [-5, 25],
    title: str = "ADE Bo20",
    save_path: Optional[str] = None,
    plot_std: bool = True,
):
    methods = list(next(iter(data_inv.values())).keys())

    max_t = max(
        [
            df["t"].max()
            for methods_data in data_inv.values()
            for df in methods_data.values()
        ]
    )
    values = {method: [] for method in methods}
    t_values = range(1, int(lim) + 1)

    for sample_id, methods_data in data_inv.items():
        for method_name, df in methods_data.items():
            df = df[df["t"] <= lim]
            df_aligned = reindex_trajectory(
                df, np.arange(1, int(lim) + 1), linear_keys=[key], nearest_keys=[]
            )
            if "ynet" in method_name:
                df[df["t"] <= 30.0]
                df_aligned = reindex_trajectory(
                    df, np.arange(1, 31), linear_keys=[key], nearest_keys=[]
                )
            values[method_name].append(df_aligned[key].values)

    means = {method: np.nanmean(vals, axis=0) for method, vals in values.items()}
    stds = {method: np.nanstd(vals, axis=0) for method, vals in values.items()}

    print("Metric: ", title)
    print({k: v[-1] for k, v in means.items()})

    # Plot
    plt.figure(figsize=(10, 6))

    for method_name in methods:
        if "ynet" in method_name:
            t_values = range(1, 31)
        else:
            t_values = range(1, int(lim) + 1)
        if method_name == "places_uniform_goal":
            legend_name = "Random goal"
        elif method_name == "reachable_states":
            legend_name = "Random reachable state"
        elif method_name == "ctmc":
            legend_name = "Random walk"
        elif method_name == "v4_deterministic":
            legend_name = "Deterministic method"
        elif method_name == "v4_instance":
            legend_name = "Instance Prediction"
        elif method_name == "v4":
            legend_name = "LP$^2$ (our method)"
        elif method_name == "v4_gt_action_sequences_instance":
            legend_name = "Given ground truth interaction sequences"
        elif method_name == "v4_gt_action_sequences_semantics":
            legend_name = "Given ground truth semantic interaction sequences"
        elif method_name == "const_vel":
            legend_name = "Constant velocity"
        elif method_name == "trajectron":
            legend_name = "Trajectron++$\\star$"
        elif method_name == "ynet":
            legend_name = "YNet$\\star$"
        else:
            legend_name = method_name
        plt.plot(
            t_values,
            means[method_name],
            label=legend_name,
            color=colors[method_name][0],
            linestyle=colors[method_name][1],
            linewidth=2,
        )

        if plot_std:
            if key == "acc":
                plt.fill_between(
                    t_values,
                    (means[method_name] - stds[method_name]).clip(min=0.0),
                    (means[method_name] + stds[method_name]).clip(max=1.0),
                    color=colors[method_name][0],
                    linestyle=colors[method_name][1],
                    linewidth=2,
                    alpha=0.2,
                )
            else:
                plt.fill_between(
                    t_values,
                    means[method_name] - stds[method_name],
                    means[method_name] + stds[method_name],
                    color=colors[method_name][0],
                    linestyle=colors[method_name][1],
                    linewidth=2,
                    alpha=0.2,
                )

    plt.xticks(np.arange(0, lim + 2, 10))
    plt.xlim([0, lim + 1])
    plt.ylim(ylim)
    plt.xlabel("time horizon [s]")
    plt.ylabel(title)
    plt.legend()
    # if plot_std:
    #     plt.title(f'{title} with Standard Deviation Shading')
    # else:
    #     plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_trajectory_env_background(
    static_tfs: pd.DataFrame,
    past_trajectory: pd.DataFrame,
    gt_future_trajectory: pd.DataFrame,
    cfg: dict,
    room_cfg: dict,
    outfile: str,
):

    img = plt.imread(cfg["floorplan_image"])
    fig, ax = plt.subplots(figsize=(8, img.shape[0] / img.shape[1] * 8))

    ax.set_xticks([])
    ax.set_yticks([])

    bottom_left_3D = unity_coordinates_to_right_hand_coordinates(
        cfg["geometry_unity"]["bottom_left"][0],
        cfg["geometry_unity"]["bottom_left"][1],
        cfg["geometry_unity"]["bottom_left"][2],
    )
    top_right_3D = unity_coordinates_to_right_hand_coordinates(
        cfg["geometry_unity"]["top_right"][0],
        cfg["geometry_unity"]["top_right"][1],
        cfg["geometry_unity"]["top_right"][2],
    )

    bottom_left_2D = cfg["image_coordinates"]["bottom_left"]
    top_right_2D = cfg["image_coordinates"]["top_right"]

    ax.imshow(img, alpha=0.7, origin="upper")

    x_past = (past_trajectory["x"] - bottom_left_3D[0]) / (
        top_right_3D[0] - bottom_left_3D[0]
    ) * (top_right_2D[0] - bottom_left_2D[0]) + bottom_left_2D[0]
    y_past = (past_trajectory["y"] - bottom_left_3D[1]) / (
        top_right_3D[1] - bottom_left_3D[1]
    ) * (top_right_2D[1] - bottom_left_2D[1]) + bottom_left_2D[1]
    y_past = img.shape[0] - y_past
    ax.plot(x_past, y_past, c="g", marker="", linewidth=8.0)

    plt.savefig(outfile.replace(".png", "_past.png"), dpi=96, bbox_inches="tight")
    plt.close("all")


def get_interactions_and_rooms(
    static_tfs: pd.DataFrame,
    past_trajectory: pd.DataFrame,
    gt_future_trajectory: pd.DataFrame,
    cfg: dict,
    bottom_left_3D: list[float],
    top_right_3D: list[float],
    bottom_left_2D: list[int],
    top_right_2D: list[int],
    img_shape: tuple[int, int],
) -> tuple[list[int], list[int], list[str], list[int]]:
    pos_x, pos_y, labels, rooms = [], [], [], []

    all_interactions = np.unique(
        np.hstack(
            [
                past_trajectory["interaction_id"].values,
                gt_future_trajectory["interaction_id"].values,
            ]
        )
    )

    for index, row in static_tfs.iterrows():
        if row.name not in all_interactions:
            continue
        if row["roomid"] == 8 and cfg["name"] == "home":
            continue
        x, y, z = unity_coordinates_to_right_hand_coordinates(
            row["x"], row["y"], row["z"]
        )
        label = row["label"] + " " + str(index)

        x_adj, y_adj = projection_3d_to_2d(
            x, y, bottom_left_3D, top_right_3D, bottom_left_2D, top_right_2D, img_shape
        )

        pos_x.append(x_adj)
        pos_y.append(y_adj)
        labels.append(label)
        rooms.append(row["roomid"])

    return pos_x, pos_y, labels, rooms


def projection_3d_to_2d(
    x_3d: Union[float, np.ndarray],
    y_3d: Union[float, np.ndarray],
    bottom_left3D: list[float],
    top_right3D: list[float],
    bottom_left2D: list[float],
    top_right2D: list[float],
    img_shape: tuple[int, int],
) -> tuple[int, int]:
    x_2d = (x_3d - bottom_left3D[0]) / (top_right3D[0] - bottom_left3D[0]) * (
        top_right2D[0] - bottom_left2D[0]
    ) + bottom_left2D[0]
    y_2d = (y_3d - bottom_left3D[1]) / (top_right3D[1] - bottom_left3D[1]) * (
        top_right2D[1] - bottom_left2D[1]
    ) + bottom_left2D[1]

    # flip y (to plt format (0,0) in top left corner):
    y_2d = img_shape[0] - y_2d

    # int:
    if isinstance(x_2d, np.ndarray):
        x_2d = x_2d.astype(int)
        y_2d = y_2d.astype(int)
    else:
        x_2d = int(x_2d)
        y_2d = int(y_2d)

    return x_2d, y_2d


def projection_2d_to_3d(
    x_2d: Union[float, np.ndarray],
    y_2d: Union[float, np.ndarray],
    bottom_left3D: list[float],
    top_right3D: list[float],
    bottom_left2D: list[float],
    top_right2D: list[float],
    img_shape: tuple[int, int],
) -> tuple[int, int]:

    # flip y (from plt format (0,0) in top left corner):
    y_2d = img_shape[0] - y_2d

    x_3d = (x_2d - bottom_left2D[0]) / (top_right2D[0] - bottom_left2D[0]) * (
        top_right3D[0] - bottom_left3D[0]
    ) + bottom_left3D[0]
    y_3d = (y_2d - bottom_left2D[1]) / (top_right2D[1] - bottom_left2D[1]) * (
        top_right3D[1] - bottom_left3D[1]
    ) + bottom_left3D[1]

    return x_3d, y_3d


def create_custom_cmap(
    cmap: colors.LinearSegmentedColormap, limits: tuple[float] = (0, 0.8)
) -> ListedColormap:
    cmapalpha = cmap(np.arange(cmap.N))
    cmapalpha[:, -1] = np.sqrt(np.linspace(limits[0], limits[1], cmap.N))
    cmapalpha = ListedColormap(cmapalpha)

    return cmapalpha
