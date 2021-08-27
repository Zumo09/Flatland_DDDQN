class AgentsController:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.action_dict = dict()

        self.update_values = [False] * n_agents

    def act(self, agent_obs, info, eps):
        for a in range(self.n_agents):
            if info['action_required'][a]:
                # If an action is require, we want to store the obs a that step as well as the action
                self.update_values[a] = True
                action_prob[actions[a]] += 1
            else:
                self.update_values[a] = False
                action = 0
            self.action_dict.update({a: action})

        return self.action_dict

    def step(self, param, param1, param2, param3, param4):
        # Update replay buffer and train agent
        for a in range(env.get_num_agents()):
            # Only update the values when we are done or when an action was taken and thus relevant information
            # is present
            if update_values[a] or done[a]:
                agent.step(agent_obs_buffer[a], agent_action_buffer[a], all_rewards[a],
                           agent_obs[a], done[a])
                cummulated_reward[a] = 0.

                agent_obs_buffer[a] = agent_obs[a].copy()
                agent_action_buffer[a] = action_dict[a]
            if next_obs[a]:
                agent_obs[a] = normalize_observation(next_obs[a], tree_depth, observation_radius=10)