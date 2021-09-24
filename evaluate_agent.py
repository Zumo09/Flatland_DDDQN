import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool

from components.dddqn_policy import DDDQNPolicy
from components.env_config import get_env_config
from components.observations import CustomObservation

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from utils.deadlock_check import check_if_all_blocked
from utils.timer import Timer
from components.bdddqn_policy import BDDDQNPolicy


def eval_policy(env_params, bootstrapped, n_eval_episodes, seed, render, allow_skipping):
    # Environment parameters
    n_agents = env_params.n_agents
    x_dim = env_params.x_dim
    y_dim = env_params.y_dim
    n_cities = env_params.n_cities
    max_rails_between_cities = env_params.max_rails_between_cities
    max_rails_in_city = env_params.max_rails_in_city

    # Malfunction and speed profiles
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=env_params.malfunction_rate,  # Rate of malfunctions
        min_duration=20,  # Minimal duration
        max_duration=50  # Max duration
    )

    # Only fast trains in Round 1
    speed_profiles = {
        1.: 1.0,  # Fast passenger train
        1. / 2.: 0.0,  # Fast freight train
        1. / 3.: 0.0,  # Slow commuter train
        1. / 4.: 0.0  # Slow freight train
    }

    # Observation parameters
    observation_tree_depth = env_params.observation_tree_depth
    observation_max_path_depth = env_params.observation_max_path_depth

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    tree_observation = CustomObservation(max_depth=observation_tree_depth, predictor=predictor)

    # Setup the environment
    env = RailEnv(
        width=x_dim, height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rails_in_city=max_rails_in_city,
        ),
        schedule_generator=sparse_schedule_generator(speed_profiles),
        number_of_agents=n_agents,
        malfunction_generator_and_process_data=malfunction_from_params(malfunction_parameters),
        obs_builder_object=tree_observation
    )

    env_renderer = None
    if render:
        env_renderer = RenderTool(env, gl="PGL")

    action_dict = dict()
    scores = []
    completions = []
    nb_steps = []
    inference_times = []
    agent_times = []
    step_times = []

    if bootstrapped:
        policy = BDDDQNPolicy(None, Namespace(num_heads=4), evaluation_mode=True)
        policy.load('./checkpoints/210923213346/1500')
        print('\nEvaluating Bootstrapped DQN\n')
    else:
        policy = DDDQNPolicy(None, None, evaluation_mode=True)
        policy.load('./checkpoints/210921124919/2000')
        print('\nEvaluating Epsilon Greedy\n')

    for episode_idx in range(n_eval_episodes):
        seed += 1

        inference_timer = Timer()
        agent_timer = Timer()
        step_timer = Timer()

        step_timer.start()
        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=seed)
        step_timer.end()

        max_steps = env._max_episode_steps

        score = 0.0

        if render:
            env_renderer.set_new_rail()

        final_step = 0
        skipped = 0

        nb_hit = 0
        agent_last_obs = {}
        agent_last_action = {}

        for step in range(max_steps - 1):
            if allow_skipping and check_if_all_blocked(env):
                skipped = max_steps - step - 1
                final_step = max_steps - 2
                n_unfinished_agents = sum(not done[idx] for idx in env.get_agent_handles())
                score -= skipped * n_unfinished_agents
                break

            agent_timer.start()
            for agent in env.get_agent_handles():
                if obs[agent] is not None and info['action_required'][agent]:
                    if agent in agent_last_obs and np.all(agent_last_obs[agent] == obs[agent]):
                        nb_hit += 1
                        action = agent_last_action[agent]

                    else:
                        norm_obs = obs[agent]

                        inference_timer.start()
                        action = policy.act(norm_obs)
                        inference_timer.end()

                    action_dict.update({agent: action})
            agent_timer.end()

            step_timer.start()
            obs, all_rewards, done, info = env.step(action_dict)
            step_timer.end()

            if render:
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=False,
                    show_predictions=False
                )

            for agent in env.get_agent_handles():
                score += all_rewards[agent]

            final_step = step

            if done['__all__']:
                break

        normalized_score = score / (max_steps * env.get_num_agents())
        scores.append(normalized_score)

        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)

        nb_steps.append(final_step)

        inference_times.append(inference_timer.get())
        agent_times.append(agent_timer.get())
        step_times.append(step_timer.get())

        skipped_text = ""
        if skipped > 0:
            skipped_text = "\t⚡ Skipped {}".format(skipped)

        hit_text = ""
        if nb_hit > 0:
            hit_text = "\t⚡ Hit {} ({:.1f}%)".format(nb_hit, (100 * nb_hit) / (n_agents * final_step))

        print(
            "☑️  Score: {:.3f} \tDone: {:4.1f}% \tNb steps: {:3d} "
            "\t🍭 Seed: {}"
            "\t🚉 Env: {:.3f}s  "
            "\t🤖 Agent: {:.3f}s (per step: {:.3f}s) \t[infer: {:.3f}s]"
            "{}{}".format(
                normalized_score,
                completion * 100.0,
                final_step,
                seed,
                step_timer.get(),
                agent_timer.get(),
                agent_timer.get() / final_step,
                inference_timer.get(),
                skipped_text,
                hit_text
            )
        )

    return scores, completions, nb_steps, agent_times, step_times


def evaluate_agents(n_evaluation_episodes, render=True):
    params = get_env_config(0)
    env_params = Namespace(**params)

    results_eps = eval_policy(env_params, False, n_evaluation_episodes, 0, render, True)
    results_boot = eval_policy(env_params, True, n_evaluation_episodes, 0, render, True)

    print("-" * 200)
    print('Results Test on Epsilon Greedy')
    scores, completions, nb_steps, agent_times, step_times = results_eps

    print("✅ Score: {:.3f} \tDone: {:.1f}% \tNb steps: {:.3f} \tAgent total: {:.3f}s (per step: {:.3f}s)".format(
        np.mean(scores),
        np.mean(completions) * 100.0,
        np.mean(nb_steps),
        np.mean(agent_times),
        np.mean(agent_times) / np.mean(nb_steps)
    ))

    print("⏲️  Agent sum: {:.3f}s \tEnv sum: {:.3f}s \tTotal sum: {:.3f}s".format(
        np.sum(agent_times),
        np.sum(step_times),
        np.sum(agent_times) + np.sum(step_times)
    ))

    print("-" * 200)
    print('Results Test on Bootstrapped')
    scores, completions, nb_steps, agent_times, step_times = results_boot

    print("✅ Score: {:.3f} \tDone: {:.1f}% \tNb steps: {:.3f} \tAgent total: {:.3f}s (per step: {:.3f}s)".format(
        np.mean(scores),
        np.mean(completions) * 100.0,
        np.mean(nb_steps),
        np.mean(agent_times),
        np.mean(agent_times) / np.mean(nb_steps)
    ))

    print("⏲️  Agent sum: {:.3f}s \tEnv sum: {:.3f}s \tTotal sum: {:.3f}s".format(
        np.sum(agent_times),
        np.sum(step_times),
        np.sum(agent_times) + np.sum(step_times)
    ))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", "--n_evaluation_episodes", help="number of evaluation episodes", default=25, type=int)
    parser.add_argument("-r", "--render", help="render env", default=True, type=bool)
    args = parser.parse_args()

    evaluate_agents(n_evaluation_episodes=args.n_evaluation_episodes, render=args.render)
