import numpy as np
from .epsilon_schedules import DecayThenFlatSchedule


class EpsilonGreedyActionSelector:

    def __init__(self, schedule=None):
        if schedule is None:
            self.schedule = DecayThenFlatSchedule(1, 0.05, 50000, decay="linear")
        else:
            self.schedule = schedule

        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, n_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        random_numbers = np.random.rand(agent_inputs.shape[0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = np.random.randint(0, n_actions, agent_inputs.shape[0])

        picked_actions = pick_random * random_actions + (1 - pick_random) * np.argmax(agent_inputs, axis=1)
        return picked_actions
