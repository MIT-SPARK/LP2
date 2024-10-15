from abc import abstractmethod, ABC
from typing import TypeVar, Any
from spark_dsg._dsg_bindings import DynamicSceneGraph as DSG
import pandas as pd


class LowLevelModuleBase(ABC):
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def process(
        self,
        scene_dsg: DSG,
        past_trajectory: pd.DataFrame,
        coarse_paths: list[list[int]],
        probabilities: list[list[float]],
    ) -> tuple[list[pd.DataFrame], float]:
        """
        Find distribution over future trajectories
        :param scene_dsg: scene graph
        :param past_trajectory: past trajectory
        :param coarse_paths: list of coarse paths
        :param probabilities: list of probabilities for each path
        :return: kde over future trajectories
        :return: prediction time horizon
        """
        return None, None

    def on_episode_init(self, scene_dsg: DSG) -> None:
        """
        Initialize module
        :param scene_dsg: scene graph
        """
        pass
