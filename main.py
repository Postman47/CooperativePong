from __future__ import annotations

import glob
import os
import time
from copy import deepcopy

import supersuit as ss
import tensorflow
from stable_baselines3 import PPO, DQN
from stable_baselines3.ppo import CnnPolicy
# from stable_baselines3.dqn import CnnPolicy
from stable_baselines3.common.vec_env import VecMonitor

from pettingzoo.butterfly import cooperative_pong_v5

def train(env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
    # Train a single model to play as each agent in an AEC environment
    print(tensorflow.config.list_physical_devices('GPU'))
    env = env_fn.parallel_env(render_mode=None, **env_kwargs)

    # Pre-process using SuperSuit
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")
    eval_env = deepcopy(env)

    eval_env = VecMonitor(eval_env)

    logdir = "logs"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Use a CNN policy if the observation space is visual
    model = PPO(
        CnnPolicy,
        eval_env,
        device="cuda",
        # buffer_size=10000,
        verbose=2,
        tensorboard_log=logdir,
        ent_coef=0.3
    )

    model.learn(total_timesteps=steps, tb_log_name="PPO")

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    # Pre-process using SuperSuit
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}
    rewards_rand = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    for i in range(num_games):
        env.reset(seed=i)
        # env.reset()
        env.action_space(env.possible_agents[0]).seed(i)
        # env.action_space(env.possible_agents[0]).seed()

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            if i >= num_games / 2:
                for a in env.agents:
                    rewards_rand[a] += env.rewards[a]
            else:
                for a in env.agents:
                    rewards[a] += env.rewards[a]

            if termination or truncation:
                break
            else:
                if i >= num_games/2:
                    act = env.action_space(agent).sample()
                else:
                    act = model.predict(obs, deterministic=True)[0]
            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: 2*rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)

    avg_reward_rand = sum(rewards_rand.values()) / len(rewards_rand.values())
    avg_reward_rand_per_agent = {
        agent: 2*rewards_rand[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward for random behaviour: {avg_reward_rand}")
    print("Avg reward for random behaviour per agent, per game: ", avg_reward_rand_per_agent)
    print("Full rewards for random behaviour: ", rewards_rand)

    return avg_reward


if __name__ == "__main__":
        env_fn = cooperative_pong_v5

        # Set vector_state to false in order to use visual observations (significantly longer training time)
        env_kwargs = dict( ball_speed=9, left_paddle_speed=12,right_paddle_speed=12,
                          cake_paddle=True, max_cycles=900, bounce_randomness=False, max_reward=100,
                          off_screen_penalty=-10)

        # Train a model (takes ~5 minutes on a laptop CPU)
        # train(env_fn, steps=2_500_000, seed=0, **env_kwargs)

        # Evaluate 10 games (takes ~10 seconds on a laptop CPU)
        # eval(env_fn, num_games=100, render_mode=None, **env_kwargs)

        eval(env_fn, num_games=2, render_mode="human", **env_kwargs)