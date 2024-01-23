import os
import sys
from torch import nn
import ray
import tensorflow as tf
import supersuit as ss
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
#from ray.rllib.env.wrappers import BaseWrapper
from ray.rllib.models import ModelCatalog,ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.models.tf.tf_modelv2 import TFModelV2


from pettingzoo.butterfly import cooperative_pong_v5


# class MLPModelV2(TFModelV2):
#     def __init__(self, obs_space, action_space, num_outputs, model_config,
#                  name="my_model"):
#         super().__init__(obs_space, action_space, num_outputs, model_config,
#                          name)
#         # Simplified to one layer.
#         input = tf.keras.layers.Input(obs_space.shape, dtype=obs_space.dtype)
#         output = tf.keras.layers.Dense(num_outputs, activation=None)
#         self.base_model = tf.keras.models.Sequential([input, output])
#         #self.register_variables(self.base_model.variables)
#     def forward(self, input_dict, state, seq_lens):
#         return self.base_model(input_dict["obs"]), []


# wraper na środowisko
# class CooperativePongWrapper(BaseWrapper, MultiAgentEnv):
#     def __init__(self, env, cooperative_reward_scale=1.0):
#         BaseWrapper.__init__(self, env)
#         self.cooperative_reward_scale = cooperative_reward_scale

#     def reset(self):
#         return self.env.reset()

#     def step(self, action_dict):
#         observations, rewards, dones, infos = self.env.step(action_dict)

#         # Custom reward shaping
#         for agent, info in infos.items():
#             if "ball_out_of_bounds" in info and info["ball_out_of_bounds"] == 0:
#                 # Positive reward for hitting the ball
#                 rewards[agent] += 1.0

#             # Additional custom reward shaping based on your criteria

#         # Cooperative reward for keeping the ball in play
#         if all(info.get("ball_out_of_bounds", 1) == 0 for info in infos.values()):
#             for agent in rewards:
#                 rewards[agent] += self.cooperative_reward_scale

#         return observations, rewards, dones, infos



def env_creator(args):
    env = cooperative_pong_v5.parallel_env(
        ball_speed=9,
        left_paddle_speed = 12,
        right_paddle_speed = 12,
        cake_paddle= False,
        max_cycles = 900,
        bounce_randomness = True
    )
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.dtype_v0(env, "float32")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    #env = ss.frame_skip_v0(env, 4)
    env = ss.frame_stack_v1(env, 4)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    return env

if __name__ == "__main__":
    ray.init()

    env_name = "cooperative_pong_v5"

    method = "PPO"
    seed = 0


    print(env_name)
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("CNNModelV2_0", CNNModelV2_paddleLeft)
    ModelCatalog.register_custom_model("CNNModelV2_1", CNNModelV2_paddleRight)

    # config = (
    #     PPOConfig()
    #     .environment(env=env_name, clip_actions=True)
    #     .training(
    #         lr=2e-5,
    #         gamma=0.99,
    #         lambda_=0.9,
    #         use_gae=True,
    #         clip_param=0.4,
    #         grad_clip=None,
    #         entropy_coeff=0.1,
    #         vf_loss_coeff=0.25,
    #         sgd_minibatch_size=64,
    #         num_sgd_iter=10,
    #     )
    #     .debugging(log_level="ERROR")
    #     .framework(framework="torch")
    #     .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    # )


    test_env = ParallelPettingZooEnv(env_creator({}))
    obs_space = test_env.observation_space
    act_space = test_env.action_space


    def gen_policy(i):
        config = {
            "model": {
                "custom_model": "CNNModelV2_{}".format(i), 
            },          
            "gamma": 0.99,
        }
        return (None, obs_space["paddle_{}".format(i)], act_space["paddle_{}".format(i)], config)


    num_agents = len(test_env.get_agent_ids())
    print(num_agents)
    policies = {"policy_{}".format(i): gen_policy(i) for i in range(num_agents)}
    # fixing obs_space and act_space

    policy_ids = list(policies.keys())

    def custom_policy_mapping_fn(agent_id, episode, worker=None):
        try:
            agent_index = int(agent_id[-1])
            return policy_ids[agent_index]
        except Exception as e:
            print(f"Error in policy_mapping_fn: {e}")
            print(agent_id)
            return None

    fullpath = os.path.abspath("ray_results/" + env_name)


    tune.run(
                "PPO",
                name="PPO",
                stop={"training_iteration": 200},
                checkpoint_freq=40,
                local_dir=fullpath,
                config={
                    # Environment specific
                    "env": env_name,
                    # General
                    "log_level": "DEBUG",#"ERROR",
                    "framework": "torch",
                    "seed": int(seed),
                    "num_cpus": 4,
                    "num_gpus": 0,
                    "num_workers": 0, # parallel workers
                    "num_envs_per_worker": 4,
                    "compress_observations": False,           
                    # 'use_critic': True,
                    'use_gae': True,
                    "lambda": 0.95,

                    "gamma": .99,

                    # "kl_coeff": 0.001,
                    # "kl_target": 1000.,
                    "kl_coeff": 0.5,
                    "clip_param": 0.3,
                    'grad_clip': None,
                    "entropy_coeff": 0.1,
                    'vf_loss_coeff': 0.25,
                    "sgd_minibatch_size": 64,
                    "num_sgd_iter": 10, # epoc
                    'rollout_fragment_length': 512,
                    "train_batch_size": 512*4,
                    'lr': 2e-05,
                    "clip_actions": True,
            
                    # Method specific
                    "multiagent": {
                        "policies": policies,
                        "policy_mapping_fn": custom_policy_mapping_fn,
                        # "policy_mapping_fn": (
                        #     lambda agent_id: policy_ids[int(agent_id[-1])]),
                    },
                },
            )
    # tune.run(
    #     "PPO",
    #     name="PPO",
    #     stop={"timesteps_total": 50000},
    #     checkpoint_freq=10,
    #     local_dir=fullpath,
    #     config=config.to_dict(),
    #)