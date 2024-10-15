import numpy as np
import pandas as pd
import os
import click
from collections.abc import Callable

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import matplotlib.colors as colors
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation

plt.rcParams.update({"font.size": 15})

from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity


import spark_dsg as dsg
from spark_dsg._dsg_bindings import DynamicSceneGraph as DSG

from lhmp.utils.dsg import Layers
from lhmp.utils.data import reindex_trajectory
from lhmp.utils.general import read_json
from lhmp.utils.visualization import *


def load_dataset_statistics(folder: str, scene: str, n_past_interactions) -> dict:
    dataset_statistics = {}
    with open(folder.format(scene, n_past_interactions), "r") as f:
        dataset_statistics = json.load(f)
    return dataset_statistics


def plot_nll_multiple_scenes(
    scenes: list[str],
    methods_nll_t: list[str],
    dataloader_function: Callable[[list[str], Optional[list[int]]], dict],
    datadirs: dict[str, str],
    color_dict: dict,
    ylimits: dict,
    lim: float = 59.0,
    plot_std: bool = False,
) -> None:

    fig, axes = plt.subplots(
        nrows=len(scenes),
        ncols=1,
        figsize=(8, 7),
        gridspec_kw={"height_ratios": [1.7, 1]},
    )

    outdir = f"output/combined/figures/NLL/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    dirs_nll_t = {
        scene: [
            os.path.join(datadirs[scene], "nll_t", method) for method in methods_nll_t
        ]
        for scene in scenes
    }
    data_nll_t_inv = {scene: dataloader_function(dirs_nll_t[scene]) for scene in scenes}

    title = "Mean Negative Log Likelihood"
    save_path = os.path.join(outdir, f"nll_stats.png")

    ax = axes[0]

    for idx, (ax, scene) in enumerate(zip(axes, scenes)):

        ax = plot_mean_and_std_multiple_scenes(
            ax,
            data_nll_t_inv[scene],
            color_dict,
            "nll",
            lim=lim,
            title="Mean Negative Log Likelihood",
            plot_std=plot_std,
        )
        ax.set_ylim(ylimits[scene])
        ax.set_xticks(np.arange(0, int(lim) + 2, 10))
        ax.tick_params(axis='x', labelsize='large')
        ax.set_xlim([0, lim])
        ax.set_ylabel("Mean NLL", size="large")
        ax.set_title(scene, size="large")

        if idx == 0:
            ax.legend(loc="lower right", fontsize="large", ncol=2)

    ax.set_xlabel("time horizon [s]", size="large")
    plt.tight_layout()

    fig.subplots_adjust(top=0.92, hspace=0.2)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close("all")


@click.command()
@click.option("--scenes", type=list[str], default=["office", "home"])
@click.option("--n_past_interactions", type=int, default=2)
@click.option(
    "--methods",
    "-m",
    type=str,
    multiple=True,
    required=True,
    default=["LP2", "LP2_instance"],
)
@click.option("--time_lim", type=float, default=60.0)
@click.option("--plot_std", type=bool, default=False)
def main(scenes, n_past_interactions, methods, time_lim, plot_std):

    ylimits = {"home": [0, 7], "office": [-5, 14]}

    dataloader_function = invert_data_structure_interaction

    color_dict = {
        "LP2": ("blue", "solid"),
        "LP2_instance": ("red", "dotted"),
    }

    plot_nll_multiple_scenes(
        scenes,
        methods,
        dataloader_function,
        datadirs={scene: f"output/{scene}/" for scene in scenes},
        color_dict=color_dict,
        ylimits=ylimits,
        lim=time_lim,
        plot_std=plot_std,
    )


if __name__ == "__main__":
    main()
