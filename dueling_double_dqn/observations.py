from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.envs.observations import TreeObsForRailEnv
from utils.observation_utils import normalize_observation


class CustomObservation(TreeObsForRailEnv):
    def __init__(self, max_depth: int, predictor: PredictionBuilder = None):
        super().__init__(max_depth, predictor=predictor)

    def normalize(self):
        return normalize_observation(self, self.max_depth, observation_radius=10)

