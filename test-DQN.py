
import matplotlib.pyplot as plt
import tensorflow as tf
import prettytensor as pt
import gym
import numpy as np
import math
import reinforcement_learning as rl

env_name = 'Breakout-v0'

rl.checkpoint_base_dir = 'SeniorProject/'
rl.update_paths(env_name=env_name)

agent = rl.Agent(env_name=env_name,
                 training=True,
                 render=True,
                 use_logging=False)
model = agent.model
replay_memory = agent.replay_memory

agent.run(num_episodes=1)

log_q_values = rl.LogQValues()
log_reward = rl.LogReward()

log_q_values.read()
log_reward.read()
