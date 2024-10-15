from abc import abstractmethod, ABC
from typing import TypeVar, Any
from spark_dsg._dsg_bindings import DynamicSceneGraph as DSG
import pandas as pd


class PlanningModuleBase(ABC):
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def on_episode_init(self, scene_graph: DSG):
        return

    @abstractmethod
    def find_paths(
        self, scene_dsg: DSG, past_trajectory: pd.DataFrame, goal_distribution: Any
    ) -> tuple[list[list[list[int]]], list[list[list[float]]]]:
        """
        Find distribution over future trajectories TODO: Define inputs and outputs (not clear)
        """
        return None, None


class TrivialPlanningModule(PlanningModuleBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def find_paths(
        self, scene_dsg: DSG, past_trajectory: pd.DataFrame, goal_distribution: Any
    ) -> tuple[list[list[list[int]]], list[list[list[float]]]]:
        return None, None
