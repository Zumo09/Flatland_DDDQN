import numpy as np

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from .tree_obs_utils import normalize_observation

TREE_DEPTH = 2
ADD_AGENT_ID = True


class CustomObservation(TreeObsForRailEnv):
    def __init__(self):
        super().__init__(max_depth=TREE_DEPTH, predictor=ShortestPathPredictorForRailEnv(30))

    def get(self, handle: int = 0):
        obs = super(CustomObservation, self).get(handle=handle)
        if obs:
            obs = normalize_observation(obs, TREE_DEPTH, observation_radius=10)
            if ADD_AGENT_ID:
                agent_id = np.zeros(self.env.get_num_agents())
                agent_id[handle] = 1.0
                obs = np.append(obs, agent_id)
        return obs
