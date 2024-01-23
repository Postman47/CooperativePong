from Enviroment import cooperative_pong_v5
#from pettingzoo.butterfly import cooperative_pong_v5
env = cooperative_pong_v5.env(
    ball_speed=9,
    left_paddle_speed = 12,
    right_paddle_speed = 12,
    cake_paddle= False,
    max_cycles = 900,
    render_mode = "human"
    )
env.reset(seed=0)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    print(reward)
    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = 0 #env.action_space(agent)[0]

    env.step(action)
env.close()