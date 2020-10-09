# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 18:22:04 2020

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
collect_steps_per_iteration = 1  # @param {type:"integer"}
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

print(agent.train_argspec)


