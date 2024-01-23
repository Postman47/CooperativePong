import os

import ray
import supersuit as ss
from PIL import Image
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn

from pettingzoo.butterfly import cooperative_pong_v5


checkpoint_path = "D:\rlProject\CooperativePong\ray_results\cooperative_pong_v5\PPO\PPO_cooperative_pong_v5_8b506_00000_0_2024-01-17_17-54-52/checkpoint_000006"

def env_creator():
    env = cooperative_pong_v5.parallel_env(
        ball_speed=9,
        left_paddle_speed = 12,
        right_paddle_speed = 12,
        cake_paddle= False,
        max_cycles = 300,
        bounce_randomness = True,
        render_mode = "human"
    )
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.dtype_v0(env, "float32")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    #env = ss.frame_skip_v0(env, 4)
    env = ss.frame_stack_v1(env, 4)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    return env


env = env_creator()
env_name = "cooperative_pong_v5"
register_env(env_name, lambda config: PettingZooEnv(env_creator()))


ray.init()

PPOagent = PPO.from_checkpoint(checkpoint_path)


reward_sum = 0
frame_list = []
i = 0
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    reward_sum += reward
    if termination or truncation:
        action = None
    else:
        action = PPOagent.compute_single_action(observation)

    env.step(action)
    i += 1
    if i % (len(env.possible_agents) + 1) == 0:
        img = Image.fromarray(env.render())
        frame_list.append(img)
env.close()


print(reward_sum)