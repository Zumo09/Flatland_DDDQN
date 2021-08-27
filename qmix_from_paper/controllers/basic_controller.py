from qmix_from_paper.models.agent__nogood import RnnAgent
from qmix_from_paper.components import EpsilonGreedyActionSelector


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, n_agents, input_shape, rnn_hidden_dim, n_actions):
        self.n_agents = n_agents
        self.agents = RnnAgent(input_shape=input_shape,
                               rnn_hidden_dim=rnn_hidden_dim,
                               n_actions=n_actions)

        self.action_selector = EpsilonGreedyActionSelector()

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs = self.agents(agent_inputs)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def load_state(self, other_mac):
        """Used to copy the weights for Double DQN"""
        pass

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass

    def _build_inputs(self, batch, t):
        """build the inputs from the observations, and add a one hot encoding for the agent_id"""
        pass
