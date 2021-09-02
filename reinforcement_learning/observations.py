import numpy as np

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from utils.observation_utils import normalize_observation


class CustomObservation(TreeObsForRailEnv):
    def __init__(self, max_depth=2, observation_radius=10, predictor=ShortestPathPredictorForRailEnv(30)):
        super().__init__(max_depth=max_depth, predictor=predictor)
        self.observation_radius = observation_radius

    def get(self, handle: int = 0):
        obs = super(CustomObservation, self).get(handle=handle)
        if obs:
            obs = normalize_observation(obs, self.max_depth, observation_radius=self.observation_radius)
            # if ADD_AGENT_ID:
            #     agent_id = np.zeros(self.env.get_num_agents())
            #     agent_id[handle] = 1.0
            #     obs = np.append(obs, agent_id)
        return obs
