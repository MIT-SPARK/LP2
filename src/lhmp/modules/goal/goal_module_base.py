from abc import abstractmethod, ABC
from spark_dsg._dsg_bindings import DynamicSceneGraph as DSG
import pandas as pd


class GoalModuleBase(ABC):
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def on_episode_init(self, scene_graph: DSG, room_labels: dict):
        return

    @abstractmethod
    def find_goals(
        self,
        scene_graph: DSG,
        room_labels: dict,
        previous_interactions: list[dict],
        past_trajectory: pd.DataFrame,
    ) -> list[list[tuple[dict, float, float]]]:
        """
        Find goals in scene graph
        :param graph: the full scene graph of the environment
        :param previous_interacions: the previous interactions with the environment's nodes
        :param past_trajectory: The past (timestamped) trajectory of the human
        :return: a list of sequences (lists) of goal nodes and their corresponding probability of being the human's future interactions
        """
        return None


class TrivialGoalModule(GoalModuleBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def find_goals(
        self,
        scene_graph: DSG,
        room_labels: dict,
        previous_interactions: list[dict],
        past_trajectory: pd.DataFrame,
    ) -> list[list[tuple[dict, float, float]]]:
        return None
