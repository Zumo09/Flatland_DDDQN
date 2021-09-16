import numpy as np
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

"""
Collection of environment-specific ObservationBuilder.
"""


class CustomObservation(TreeObsForRailEnv):
    def __init__(self, max_depth=2, predictor=ShortestPathPredictorForRailEnv(30)):
        super().__init__(max_depth=max_depth, predictor=predictor)

    def get(self, handle: int = 0):
        obs = super(CustomObservation, self).get(handle=handle)
        obs = self.normalize_observation(obs)
        return obs

    @staticmethod
    def sigmoid(array):
        return 1 / (1 + np.exp(-array))

    @staticmethod
    def _extract_node_info(node) -> np.ndarray:
        data = np.zeros(11)

        data[0] = node.dist_min_to_target

        data[1] = node.dist_own_target_encountered
        data[2] = node.dist_other_target_encountered
        data[3] = node.dist_other_agent_encountered
        data[4] = node.dist_potential_conflict
        data[5] = node.dist_unusable_switch
        data[6] = node.dist_to_next_branch

        data[7] = node.num_agents_same_direction
        data[8] = node.num_agents_opposite_direction
        data[9] = node.num_agents_malfunctioning
        data[10] = node.speed_min_fractional

        return data

    def _flatten_tree(self, node, current_tree_depth: int) -> np.ndarray:
        if node is None or node == -np.inf:
            remaining_depth = self.max_depth - current_tree_depth
            # reference: https://stackoverflow.com/questions/515214/total-number-of-nodes-in-a-tree-data-structure
            num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
            return np.full(num_remaining_nodes * 11, -np.inf)

        data = self._extract_node_info(node)

        if not node.childs:
            return data

        for direction in TreeObsForRailEnv.tree_explored_actions_char:
            sub_data = self._flatten_tree(node.childs[direction], current_tree_depth + 1)
            data = np.concatenate((data, sub_data))

        return data

    def normalize_observation(self, observation):
        """
        This function normalizes the observation used by the RL algorithm
        """
        data = self._flatten_tree(observation, 0)
        data = self.sigmoid(data)
        return data
