# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 16:15:56 2020

@author: Percy
"""
from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
#
import tensorflow as tf

from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.policies import ou_noise_policy
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.environments import suite_pybullet
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.experimental.train.utils import spec_utils
from tf_agents.agents.ddpg import critic_network
from tf_agents.utils import common
from Networks import ActorNetwork
tf.compat.v1.enable_v2_behavior()


###################HYPERPARAMETERS##############
num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"} 
collect_steps_per_iteration = 64  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}



#####LOAD ENVIRONMENT #####
env_name = "MinitaurBulletEnv-v0"
env = suite_pybullet.load(env_name)
env.reset()
PIL.Image.fromarray(env.render())


####TWO ENV instantiated. One for Train, One for Eval ######
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)
#Converts Numpy Arrays to Tensors, so they are compatible with Tensorflow agents and policies
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

observation_spec, action_spec, time_step_spec = (spec_utils.get_tensor_specs(train_env))
#print(observation_spec)
#######Networks#####
#conv_layer_params = [(32,3,3),(32,3,3),(32,3,3)]
conv_layer_params = None
fc_layer_params=(200, 200)
kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1. / 3., mode='fan_in', distribution='uniform')
final_layer_initializer = tf.keras.initializers.RandomUniform(
        minval=-0.0003, maxval=0.0003)
actor_net = ActorNetwork(observation_spec,
               action_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params= conv_layer_params,
               fc_layer_params=(200, 200),
               dropout_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               enable_last_layer_zero_initializer=False,
               name='ActorNetwork')

critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_conv_layer_params=conv_layer_params,
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=fc_layer_params,
        kernel_initializer=kernel_initializer,
        last_kernel_initializer=final_layer_initializer)

target_actor_net = ActorNetwork(observation_spec,
               action_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params= conv_layer_params,
               fc_layer_params=(200, 200),
               dropout_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               enable_last_layer_zero_initializer=False,
               name='TargetActorNetwork')

target_critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_conv_layer_params=conv_layer_params,
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=fc_layer_params,
        kernel_initializer=kernel_initializer,
        last_kernel_initializer=final_layer_initializer)

#########AGENTS######

actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_step_counter = tf.Variable(0)

agent = ddpg_agent.DdpgAgent(
    time_step_spec= time_step_spec,
    action_spec= action_spec,
    actor_network= actor_net,
    critic_network= critic_net,
    actor_optimizer=actor_optimizer,
    critic_optimizer= critic_optimizer,
    ou_stddev= 0.15,
    ou_damping= 0.2,
    target_actor_network= target_actor_net,
    target_critic_network= target_critic_net,
    target_update_tau= 0.001,
    target_update_period= 1,
    dqda_clipping= None,
    td_errors_loss_fn= None,
    gamma= 0.99,
    reward_scale_factor= None,
    gradient_clipping= None,
    debug_summaries= None,
    summarize_grads_and_vars= False,
    train_step_counter= train_step_counter,
    name= "Agent"
)


agent.initialize()

####REPLAY BUFFER####
##NOTE: Data Spec describes what goes in. Can't confirm what is being stored in ReplayBuffer!##
##May  need table...##
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

dataset = replay_buffer.as_dataset(
    sample_batch_size=batch_size,num_steps=2)
#Not sure why I need a lambda here, just seems like it returns the dataset?##
#experience_dataset_fn = lambda: dataset

########POLICY#######
tf_eval_policy = agent.policy
tf_collect_policy = agent.collect_policy
eval_policy = tf_eval_policy
collect_policy = tf_collect_policy
#eval_policy = ou_noise_policy.OUNoisePolicy(tf_eval_policy,0.15,0.2)
#collect_policy = ou_noise_policy.OUNoisePolicy(tf_collect_policy,0.15,0.2)
#random_policy = OUNoisePolicy(train_env.time_step_spec(),
##                                                train_env.action_spec())
#
#example_environment = tf_py_environment.TFPyEnvironment(
#    suite_gym.load('CartPole-v0'))
#
#time_step = example_environment.reset()
#
#random_policy.action(time_step)

#####ACTORS######
#initial_collect_actor = actor.Actor()



####DATA COLLECTION ####
print("we got to 2")
def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

compute_avg_return(eval_env, eval_policy, num_eval_episodes)


def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)
  # Add trajectory to the replay buffer
  buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)
print("we got to 3")
#collect_data(train_env, collect_policy, replay_buffer, steps=100)

# This loop is so common in RL, that we provide standard implementations. 
# For more details see the drivers module.
# https://www.tensorflow.org/agents/api_docs/python/tf_agents/drivers


iterator = iter(dataset)

#####TRAIN AND EXECUTE #####
#try:
#  %%time
#except:
#  pass

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
#agent.train = common.function(agent.train)

# Reset the train step
#agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
print(avg_return)
returns = [avg_return]
print("we got to 4")
for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  for _ in range(collect_steps_per_iteration):
    collect_step(train_env, agent.collect_policy, replay_buffer)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train = agent.train(experience)
  print("Agent Trained")
  train_loss = train.loss
  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)


####PLOT#####
    
iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=250)




