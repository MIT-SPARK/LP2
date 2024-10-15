import spark_dsg as dsg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from typing import Optional
import math

from sklearn.neighbors import KernelDensity
from scipy.interpolate import interp1d
from scipy.spatial.distance import jensenshannon

from spark_dsg._dsg_bindings import NodeSymbol, DynamicSceneGraph

from lhmp.utils.data import (
    reindex_trajectory,
    GoalSequences,
    GoalSequence,
    CoarsePaths,
    SmoothPath,
    SmoothPaths,
)

from dataclasses import asdict


class Evaluation:
    def __init__(
        self,
        time_horizon: float,
        alpha: float,
        beta: float,
        eval_timestep: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        self.verbose = True

        self.k_samples: int = 10
        self.time_horizon = time_horizon

        if "log_likelihood_lower_bound" in kwargs:
            self.log_likelihood_lower_bound = kwargs["log_likelihood_lower_bound"]
        else:
            self.log_likelihood_lower_bound = -20

        if "top_n_accuracy_N_id" in kwargs:
            self.N_top_N_acc_id = kwargs["top_n_accuracy_N_id"]

        if "top_n_accuracy_N_semantic" in kwargs:
            self.N_top_N_acc_semantic = kwargs["top_n_accuracy_N_semantic"]

        if "bandwidth" in kwargs:
            self.bandwidth = kwargs[
                "bandwidth"
            ]  # loaded from method config if given, else scene config
        else:
            self.bandwidth = "scott"

        self.alpha = alpha
        self.beta = beta

        self.eval_timestep = eval_timestep

        if "epsilon_th_steady_state" in kwargs:
            self.epsilon_th_steady_state = kwargs["epsilon_th_steady_state"]
        else:
            self.epsilon_th_steady_state = 0.05

    def evaluate_trajectory(
        self,
        trajectory_data: dict,
        evaluate_accuracy=True,
        evaluate_nll=True,
        evaluate_ade_fde=True,
        evaluate_interaction_prediction=True,
        evaluate_room_prediction=True,
        evaluate_time_horizon_steady_state=True,
        N_BoN: int = 10,
    ):
        """
        Evaluate all metrics for a single ground truth trajectory
        """
        new_timesteps = np.arange(
            trajectory_data["gt_future_trajectory"]["t"].iloc[0],
            min(
                trajectory_data["gt_future_trajectory"]["t"].iloc[-1], self.time_horizon
            ),
            self.eval_timestep,
        )
        if len(new_timesteps) == 0:
            return None, None, None, None, None

        # time from start of the trajectory (including past) until end of the prediction
        if self.alpha and self.beta:
            time_horizon_future = self.time_horizon * (self.alpha + self.beta)
        else:
            time_horizon_future = min(
                self.time_horizon, trajectory_data["gt_future_trajectory"]["t"].iloc[-1]
            )

        # time from start of the prediction until end of the prediction
        if self.beta:
            prediction_horizon = (
                trajectory_data["gt_future_trajectory"].iloc[-1]["t"] * self.beta
            )
        else:
            prediction_horizon = min(
                trajectory_data["gt_future_trajectory"].iloc[-1]["t"], 60.0
            )

        gt_trajectory_reindexed = reindex_trajectory(
            trajectory_data["gt_future_trajectory"],
            new_timesteps,
            linear_keys=["x", "y", "z"],
            nearest_keys=["interaction_id", "room_id"],
        )
        paths_reindexed = [
            reindex_trajectory(
                path.path,
                new_timesteps,
                linear_keys=["x", "y"],
                nearest_keys=["room_id"],
            )
            for path in trajectory_data["pred_trajectories_interpolated"]
        ]

        if evaluate_accuracy:
            acc_at_tresholds_BoN = self.trajectory_accuracy_at_thresholds_BoN(
                paths_reindexed,
                gt_trajectory_reindexed,
                trajectory_data["trajectory_probabilities"],
                thresholds=[0.5, 1.0, 2.0, 4.0, 8.0],
                time_horizon_future=time_horizon_future,
                N=N_BoN,
            )

            survival_rate_BoN_df, survival_time = self.compute_survival_rate_BoN(
                paths_reindexed,
                gt_trajectory_reindexed,
                trajectory_data["trajectory_probabilities"],
                threshold=5.0,
                time_horizon_future=time_horizon_future,
                N=N_BoN,
            )
        else:
            acc_at_tresholds_BoN = None
            survival_rate_BoN_df = None
            survival_time = None

        if evaluate_nll:
            res_nll = self.negative_log_likelihood(
                paths_reindexed,
                gt_trajectory_reindexed,
                trajectory_data["kdes"],
                eval_timesteps=new_timesteps,
                prediction_horizon=prediction_horizon,
            )
        else:
            res_nll = None

        if evaluate_ade_fde:
            res_ade_fde_at_t = self.sequence_prediction_ADE_FDE_BoN(
                paths_reindexed,
                trajectory_data["trajectory_probabilities"],
                gt_trajectory_reindexed,
                prediction_horizon=prediction_horizon,
                N=N_BoN,
            )
        else:
            res_ade_fde_at_t = None

        if evaluate_interaction_prediction and (
            trajectory_data["future_trajectory_info"] is not None
        ):
            interaction_instance_prediction_likelihood_at_n = (
                self.compute_interaction_prediction_likelihood(
                    trajectory_data["goal_distribution"],
                    trajectory_data["future_trajectory_info"]["sequence"],
                    trajectory_data["gt_future_trajectory"],
                    key="id",
                )
            )

            interaction_semantics_prediction_likelihood_at_n = (
                self.compute_interaction_prediction_likelihood(
                    trajectory_data["goal_distribution"],
                    trajectory_data["future_trajectory_info"]["sequence"],
                    trajectory_data["gt_future_trajectory"],
                    key="semantic_label",
                )
            )
        else:
            interaction_instance_prediction_likelihood_at_n = None
            interaction_semantics_prediction_likelihood_at_n = None

        if evaluate_interaction_prediction and (
            trajectory_data["future_trajectory_info"] is not None
        ):
            prediction_top_n_accuracy_by_id = (
                self.get_prediction_top_n_accuracys_by_key(
                    trajectory_data["goal_distribution"],
                    trajectory_data["future_trajectory_info"]["sequence"],
                    trajectory_data["gt_future_trajectory"],
                    key="id",
                    N=self.N_top_N_acc_id,
                )
            )

            prediction_top_n_accuracy_by_semantic_class = (
                self.get_prediction_top_n_accuracys_by_key(
                    trajectory_data["goal_distribution"],
                    trajectory_data["future_trajectory_info"]["sequence"],
                    trajectory_data["gt_future_trajectory"],
                    key="semantic_label",
                    N=self.N_top_N_acc_semantic,
                )
            )
        else:
            prediction_top_n_accuracy_by_id = None
            prediction_top_n_accuracy_by_semantic_class = None

        if evaluate_room_prediction:
            room_prediction_accs = self.room_prediction_accuracy_BoN(
                gt_trajectory_reindexed,
                paths_reindexed,
                trajectory_data["trajectory_probabilities"],
                prediction_horizon=prediction_horizon,
                N=N_BoN,
            )
        else:
            room_prediction_accs = None

        if (
            evaluate_time_horizon_steady_state
            and "kdes" in trajectory_data
            and "steady_state" in trajectory_data
            and trajectory_data["kdes"] is not None
            and trajectory_data["steady_state"] is not None
        ):
            steady_state_distance_js = (
                self.evaluate_time_horizon_steady_state_jensen_shannon(
                    trajectory_data["kdes"],
                    trajectory_data["steady_state"],
                    trajectory_data["past_trajectory"]["t"].iloc[-1],
                )
            )
            steady_state_distance_tv = (
                self.evaluate_time_horizon_steady_state_total_variation(
                    trajectory_data["kdes"],
                    trajectory_data["steady_state"],
                    trajectory_data["past_trajectory"]["t"].iloc[-1],
                )
            )
        else:
            steady_state_distance_js = None
            steady_state_distance_tv = None

        return (
            res_nll,
            res_ade_fde_at_t,
            interaction_instance_prediction_likelihood_at_n,
            interaction_semantics_prediction_likelihood_at_n,
            prediction_top_n_accuracy_by_id,
            prediction_top_n_accuracy_by_semantic_class,
            room_prediction_accs,
            steady_state_distance_js,
            steady_state_distance_tv,
        )

    def median_absolute_trajectory_error_BoN(
        self,
        paths_reindexed: list[pd.DataFrame],
        gt_trajectory_reindexed: pd.DataFrame,
        time_horizon_future: np.float64 = 60.0,
    ) -> pd.DataFrame:
        """
        compute the median absolute trajectory error for the best of N trajectories
        """
        pass

    def trajectory_accuracy_at_thresholds_BoN(
        self,
        paths_reindexed: list[pd.DataFrame],
        gt_trajectory_reindexed: pd.DataFrame,
        trajectory_probabilities: list[float],
        thresholds: list[float],
        time_horizon_future: np.float64 = 60.0,
        N=10,
    ) -> pd.DataFrame:
        """
        Compute the position accuracy at different thresholds. Average over thresholds.
        """
        indices = np.argsort(trajectory_probabilities)[::-1][:N]
        top_k_paths_reindexed = [paths_reindexed[i] for i in indices]

        accs = []
        for path in top_k_paths_reindexed:
            path_accs = []
            for threshold in thresholds:
                path_accs.append(
                    self.trajectory_accuracy_single_threshold(
                        path,
                        gt_trajectory_reindexed,
                        threshold,
                        time_horizon_future=time_horizon_future,
                    )
                )
            accs.append(np.mean(path_accs))

        return np.max(accs)

    def trajectory_accuracy_single_threshold(
        self,
        predicted_trajectory: pd.DataFrame,
        gt_trajectory_reindexed: pd.DataFrame,
        threshold: float,
        time_horizon_future: np.float64 = 60.0,
    ) -> pd.DataFrame:
        """
        computes the accuracy of a single predicted trajectory against a threshold
        """
        assert np.all(
            gt_trajectory_reindexed["t"] == predicted_trajectory["t"]
        ), "Reindexing failed, timesteps of trajectories do not match"
        distances = np.linalg.norm(
            np.subtract(
                gt_trajectory_reindexed[
                    gt_trajectory_reindexed["t"] <= time_horizon_future
                ][["x", "y"]].to_numpy(),
                predicted_trajectory[
                    gt_trajectory_reindexed["t"] <= time_horizon_future
                ][["x", "y"]].to_numpy(),
            ),
            axis=1,
        )
        accuracy = (distances <= threshold).sum() / len(distances)
        return accuracy

    def compute_survival_rate_BoN(
        self,
        paths_reindexed: list[pd.DataFrame],
        gt_trajectory_reindexed: pd.DataFrame,
        trajectory_probabilities: list[float],
        threshold: float,
        time_horizon_future: np.float64 = 180.0,
        N=10,
    ) -> pd.DataFrame:
        """
        Compute the survival rate of trajectories. (portion of the trajectory as fraction of time horizon before the difference of trajectory to ground truth is larger than threshold)
        """
        indices = np.argsort(trajectory_probabilities)[::-1][:N]
        top_k_paths_reindexed = [paths_reindexed[i] for i in indices]

        survival_rate_df = pd.DataFrame()
        survival_time = 0.0
        for path in top_k_paths_reindexed:
            survival_rate_df_new = self.compute_survival_single_trajectory(
                path,
                gt_trajectory_reindexed,
                threshold,
                time_horizon_future=time_horizon_future,
            )
            if np.any(survival_rate_df_new["survival_rate"] == 1.0):
                survival_time_new = round(
                    survival_rate_df_new[survival_rate_df_new["survival_rate"] == 1.0][
                        "t"
                    ].iloc[-1]
                    - gt_trajectory_reindexed["t"].iloc[0],
                    5,
                )
            else:
                survival_time_new = 0.0
            if survival_time_new > survival_time:
                survival_time = survival_time_new
                survival_rate_df = survival_rate_df_new

        return survival_rate_df, survival_time

    def compute_survival_single_trajectory(
        self,
        predicted_trajectory: pd.DataFrame,
        gt_trajectory_reindexed: pd.DataFrame,
        threshold: float,
        time_horizon_future: np.float64 = 60.0,
    ) -> pd.DataFrame:
        """
        TODO: need to rename
        Compute the survival rate of a single trajectory (portion of the trajectory as fraction of time horizon before the difference of trajectory to ground truth is larger than threshold).
        """

        assert np.all(
            gt_trajectory_reindexed["t"] == predicted_trajectory["t"]
        ), "Reindexing failed, timesteps of trajectories do not match"
        distances = np.linalg.norm(
            np.subtract(
                gt_trajectory_reindexed[
                    gt_trajectory_reindexed["t"] <= time_horizon_future
                ][["x", "y"]].to_numpy(),
                predicted_trajectory[
                    gt_trajectory_reindexed["t"] <= time_horizon_future
                ][["x", "y"]].to_numpy(),
            ),
            axis=1,
        )
        if np.all(distances <= threshold):
            survival_rates = [1.0] * len(distances)
        elif np.all(distances > threshold):
            survival_rates = [0.0] * len(distances)
        else:
            first_index = np.where(distances >= threshold)[0][0]
            survival_rates = [1.0] * (first_index + 1) + [
                gt_trajectory_reindexed.iloc[first_index]["t"] / t
                for idx, t in enumerate(
                    gt_trajectory_reindexed["t"].iloc[first_index + 1 :]
                )
            ]
        return pd.DataFrame(
            np.c_[gt_trajectory_reindexed["t"], survival_rates],
            columns=["t", "survival_rate"],
        )

    def negative_log_likelihood(
        self,
        paths_reindexed: list[pd.DataFrame],
        gt_trajectory_reindexed: pd.DataFrame,
        kdes: Optional[list[dict]],
        eval_timesteps: np.ndarray,
        prediction_horizon: np.float64 = 60.0,
    ) -> pd.DataFrame:
        pred_trajectories_reindexed = [
            np.array(path[["x", "y"]]).T for path in paths_reindexed
        ]

        pred_trajectories_reindexed = np.array(pred_trajectories_reindexed)
        gt_trajectory_reindexed_np = np.array(gt_trajectory_reindexed[["x", "y"]])

        kde_ll = 0.0

        nlls = []
        nlls_cum = []
        times = []

        t0 = gt_trajectory_reindexed["t"].iloc[0]

        for idx, timestep in enumerate(eval_timesteps):
            try:
                if kdes is not None:
                    kde = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth)
                    kde.fit(
                        kdes[idx]["samples"],
                        sample_weight=kdes[idx]["sample_weights"]
                        / sum(kdes[idx]["sample_weights"]),
                    )
                    assert math.isclose(
                        kdes[idx]["t"], timestep, rel_tol=1e-4
                    ), "KDE time does not match timestep"
                else:
                    kde = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth)
                    samples = pred_trajectories_reindexed[:, :, idx]
                    kde.fit(samples)

                assert math.isclose(
                    timestep, gt_trajectory_reindexed["t"].iloc[idx], rel_tol=1e-4
                ), "Timestep does not match ground truth trajectory timestep"
                logpd = np.clip(
                    kde.score(np.expand_dims(gt_trajectory_reindexed_np[idx], axis=0)),
                    a_min=self.log_likelihood_lower_bound,
                    a_max=np.inf,
                )
                kde_ll += logpd
                nlls_cum.append(-kde_ll / (idx + 1))
                nlls.append(-logpd)
                times.append(round(gt_trajectory_reindexed["t"].iloc[idx] - t0, 4))

            except Exception as e:
                print("Exception in nll computation:\n", e)
                kde_ll = np.nan

        times = np.array(times)
        nlls = np.array(nlls)
        nlls_cum = np.array(nlls_cum)
        print_time = 60.0
        if times[-1] < 60.0:
            print_time = times[-1]
            nll_eval = nlls_cum[-1]
        else:
            nll_eval = nlls_cum[times >= 60.0][0]

        print(f"NLL at {print_time}s: ", nll_eval)
        return pd.DataFrame(
            np.c_[[nlls, nlls_cum, times, times <= prediction_horizon]].T,
            columns=["nll", "nll_cum", "t", "t_pred"],
        )

    def room_prediction_accuracy_BoN(
        self,
        gt_trajectory_reindexed: pd.DataFrame,
        paths_reindexed: list[pd.DataFrame],
        trajectory_probabilities: list[float],
        prediction_horizon: float = 60.0,
        N: int = 10,
    ) -> pd.DataFrame:
        """
        Compute the best of N room prediction accuracy.
        """
        indices = np.argsort(trajectory_probabilities)[::-1][:N]
        top_k_paths_reindexed = [paths_reindexed[i] for i in indices]

        true_positives = 0
        total = 0
        times = []
        accs_cumulative = []

        t0 = gt_trajectory_reindexed["t"].iloc[0]

        acc_cumulative = -np.inf
        best_path = None
        for path in top_k_paths_reindexed:
            true_positives = (
                path[gt_trajectory_reindexed["t"] - t0 <= prediction_horizon]["room_id"]
                == gt_trajectory_reindexed[
                    gt_trajectory_reindexed["t"] - t0 <= prediction_horizon
                ]["room_id"]
            ).sum()
            total = len(
                gt_trajectory_reindexed[
                    gt_trajectory_reindexed["t"] - t0 <= prediction_horizon
                ]
            )
            acc_cumulative_new = true_positives / total
            if acc_cumulative_new > acc_cumulative:
                acc_cumulative = acc_cumulative_new
                best_path = path

        accs_cumulative = np.cumsum(
            best_path["room_id"] == gt_trajectory_reindexed["room_id"]
        ) / np.arange(1, len(best_path) + 1)
        times = round(gt_trajectory_reindexed["t"] - t0, 5)

        return pd.DataFrame(
            np.c_[[accs_cumulative, times, times <= prediction_horizon]].T,
            columns=["acc", "t", "t_pred"],
        )

    def sequence_prediction_ADE_FDE_BoN(
        self,
        predicted_trajectories: list[pd.DataFrame],
        trajectory_probabilities: list[float],
        gt_future_trajectory: pd.DataFrame,
        prediction_horizon: float = 60.0,
        N: int = 20,
    ) -> pd.DataFrame:
        """
        Compute the best of N ADE metric for sequence prediction.
        """
        indices = np.argsort(trajectory_probabilities)[::-1][:N]

        top_k_paths = [predicted_trajectories[i] for i in indices]
        assert len(indices) <= N, "Number of trajectories does not match N"
        assert len(top_k_paths) <= N, "Number of trajectories does not match N"

        fde_at_t = []
        ade_at_t = []
        t0 = gt_future_trajectory["t"].iloc[0]
        times = gt_future_trajectory["t"].to_numpy() - t0

        t0 = gt_future_trajectory["t"].iloc[0]
        distances = np.array([])
        ade_end = np.inf
        fde_end = np.inf
        for path in top_k_paths:
            distances_path = self.compute_distances(path, gt_future_trajectory)
            ade_end_new = np.mean(
                distances_path[gt_future_trajectory["t"] - t0 <= prediction_horizon]
            )
            fde_end_new = distances_path.iloc[-1]
            if ade_end_new < ade_end:
                ade_end = ade_end_new
                distances = distances_path
                fde_at_t = distances_path

        ade_at_t = np.cumsum(distances) / np.arange(1, len(distances) + 1)

        if len(ade_at_t) <= 60:
            print_time = times[-1]
            print(f"Bo{N} ADE at {print_time}s: ", ade_at_t.iloc[-1])
        else:
            print(f"Bo{N} ADE at {60}s: ", ade_at_t[60])

        return pd.DataFrame(
            np.c_[[ade_at_t, fde_at_t, times, times <= prediction_horizon]].T,
            columns=["ade", "fde", "t", "t_pred"],
        )

    def get_prediction_top_n_accuracys_by_key(
        self,
        predicted_interaction_sequences: GoalSequences,
        future_interaction_sequence: list[dict],
        gt_future_trajectory: pd.DataFrame,
        key: str,
        N: int,
    ) -> float:
        """
        Compute the top_n_accuracy of the interaction prediction.
        """
        accs = {}
        ignore_first = False
        if gt_future_trajectory.iloc[0]["interaction_id"] != -1:
            ignore_first = True
        for n in range(ignore_first, len(future_interaction_sequence)):
            if (
                predicted_interaction_sequences is not None
                and len(predicted_interaction_sequences.sequences[0].goals) - 1 > n
            ):
                predictions_at_n = [
                    {
                        key: getattr(i.goals[n + 1], key),
                        "probability": np.product([j.probability for j in i.goals]),
                    }
                    for i in predicted_interaction_sequences.sequences
                ]
                unique_predictions = [
                    int(j) if key == "id" else str(j)
                    for j in np.unique([i[key] for i in predictions_at_n])
                ]
                prediction_likelihood_by_key = {
                    i: float(
                        np.sum(
                            [j["probability"] for j in predictions_at_n if j[key] == i]
                        )
                    )
                    for i in unique_predictions
                }
                top_k_predictions = sorted(
                    prediction_likelihood_by_key,
                    key=lambda x: prediction_likelihood_by_key[x],
                    reverse=True,
                )[:N]
                accs[int(n + (not ignore_first))] = (
                    1 if future_interaction_sequence[n][key] in top_k_predictions else 0
                )
            else:
                accs[int(n + (not ignore_first))] = None

        return accs

    def compute_interaction_prediction_likelihood(
        self,
        predicted_interaction_sequences: GoalSequences,
        future_interaction_sequence: list[dict],
        gt_future_trajectory: pd.DataFrame,
        key: str,
    ):
        node_likelihood_at_n = {}
        ignore_first = False
        if gt_future_trajectory.iloc[0]["interaction_id"] != -1:
            ignore_first = True
        for n in range(int(ignore_first), len(future_interaction_sequence)):
            if (
                predicted_interaction_sequences is not None
                and len(predicted_interaction_sequences.sequences[0].goals) - 1 > n
            ):  # first node in predicted_interaction_sequences is root node
                node_likelihood_at_n[n + 1] = (
                    self.interaction_prediction_likelihood_at_step_n(
                        predicted_interaction_sequences,
                        future_interaction_sequence,
                        n,
                        key=key,
                    )
                )
            else:
                node_likelihood_at_n[n + 1] = None

        return pd.DataFrame(
            node_likelihood_at_n.values(), index=node_likelihood_at_n.keys()
        )

    def interaction_prediction_likelihood_at_step_n(
        self,
        goal_sequences: GoalSequences,
        future_interaction_sequence: list,
        n: int = 1,
        key: str = "id",
    ):
        gt_at_n = future_interaction_sequence[n][key]

        probability = 0.0

        for prediction in goal_sequences:  # edit
            if len(prediction.goals) < n:
                continue
            if getattr(prediction.goals[n + 1], key) == gt_at_n:
                probability += prediction.probability

        return probability / sum([i.probability for i in goal_sequences])

    def compute_distances(self, trajectory: pd.DataFrame, gt_trajectory: pd.DataFrame):
        """
        Compute the average displacement error between a predicted trajectory and a ground truth trajectory.
        """
        assert np.all(
            trajectory["t"] == gt_trajectory["t"]
        ), "Reindexing failed, timesteps of trajectories do not match"
        distances = np.sqrt(
            (gt_trajectory["x"] - trajectory["x"]) ** 2
            + (gt_trajectory["y"] - trajectory["y"]) ** 2
        )

        return distances

    def evaluate_time_horizon_steady_state_total_variation(
        self, kdes: dict, steady_state: np.ndarray, t0: float
    ) -> float:
        """
        Evaluate the time it takes until the output probability is within epsilon of the steady state.
        """
        steady_state = np.abs(steady_state)

        times = [round(kde["t"] - t0, 3) for kde in kdes]
        distances = []

        for idx, kde in enumerate(kdes):
            state_dist = kde["state_dist"]
            distance = np.sum(np.abs(state_dist - steady_state)) / 2
            distances.append(distance)
            if distance < self.epsilon_th_steady_state:
                print(f"Total Variation distance at {round(kde['t'],3)}: ", distance)

        return pd.DataFrame(np.c_[[distances, times]].T, columns=["tv_distance", "t"])

    def evaluate_time_horizon_steady_state_jensen_shannon(
        self, kdes: dict, steady_state: np.ndarray, t0: float
    ) -> float:
        """
        Evaluate the time it takes until the output probability is within epsilon of the steady state.
        """
        steady_state = np.abs(steady_state)

        times = [round(kde["t"] - t0, 3) for kde in kdes]
        distances = []

        for idx, kde in enumerate(kdes):
            state_dist = kde["state_dist"]
            distance = jensenshannon(state_dist, steady_state)
            distances.append(distance)
            if distance < self.epsilon_th_steady_state:
                print(f"Jensen-Shannon distance at {round(kde['t'],3)}: ", distance)

        return pd.DataFrame(np.c_[[distances, times]].T, columns=["js_distance", "t"])
