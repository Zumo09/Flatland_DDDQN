from env.env_generator import FlatlandEnv
import numpy as np


def main():
    env = FlatlandEnv(n_agents=4, episode_limit=50, render=False)
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    n_episodes = 20

    for e in range(n_episodes):
        _, _ = env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            actions = []
            for agent_id in range(n_agents):
                action = np.random.choice(n_actions)
                actions.append(action)

            reward, terminated, info = env.step(actions)
            episode_reward += reward
            # print(env.episode_t, reward)

        if not terminated:
            episode_reward += 100

        print("Total reward in episode {} = {}".format(e, episode_reward))


if __name__ == '__main__':
    main()
