# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 16:15:56 2020

@author: Percy
"""
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import tempfile

#
import tensorflow as tf
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.experimental.train.utils import train_utils
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.environments import suite_pybullet
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import random_tf_policy
from tf_agents.experimental.train.utils import spec_utils
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import actor_network
from tf_agents.utils import common
from tf_agents.policies import py_tf_eager_policy
tf.compat.v1.enable_v2_behavior()


tempdir = tempfile.gettempdir()

###################HYPERPARAMETERS##################
num_iterations = 200 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"} 
collect_steps_per_iteration = 64  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 10  # @param {type:"integer"}

policy_save_interval = 5000 # @param {type:"integer"}

#####LOAD ENVIRONMENT #####
env_name = "MinitaurBulletEnv-v0"
env = suite_pybullet.load(env_name)

####TWO ENV instantiated. One for Train, One for Eval ######
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)
#Converts Numpy Arrays to Tensors, so they are compatible with Tensorflow agents and policies
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

time_step = env.reset()
observation_spec, action_spec, time_step_spec = (spec_utils.get_tensor_specs(train_env))



#######Networks#####
#conv_layer_params = [(32,3,3),(32,3,3),(32,3,3)]
conv_layer_params = None
fc_layer_params=(200, 100)
kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1. / 3., mode='fan_in', distribution='uniform')
final_layer_initializer = tf.keras.initializers.RandomUniform(
        minval=-0.0003, maxval=0.0003)

actor_net = actor_network.ActorNetwork(observation_spec,
               action_spec,
               conv_layer_params= conv_layer_params,
               fc_layer_params=fc_layer_params,
               dropout_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               kernel_initializer = kernel_initializer,
               last_kernel_initializer=final_layer_initializer,
               name='ActorNetwork')

critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_conv_layer_params=conv_layer_params,
        observation_fc_layer_params=(200,),
        action_fc_layer_params=None,
        joint_fc_layer_params=(100,),
        kernel_initializer=kernel_initializer,
        last_kernel_initializer=final_layer_initializer)

#target_actor_net = ActorNetwork(observation_spec,
#               action_spec,
#               preprocessing_layers=None,
#               preprocessing_combiner=None,
#               conv_layer_params= conv_layer_params,
#               fc_layer_params=(200, 200),
#               dropout_layer_params=None,
#               activation_fn=tf.keras.activations.relu,
#               enable_last_layer_zero_initializer=False,
#               name='TargetActorNetwork')
#
#target_critic_net = critic_network.CriticNetwork(
#        (observation_spec, action_spec),
#        observation_conv_layer_params=conv_layer_params,
#        observation_fc_layer_params=None,
#        action_fc_layer_params=None,
#        joint_fc_layer_params=fc_layer_params,
#        kernel_initializer=kernel_initializer,
#        last_kernel_initializer=final_layer_initializer)

#########AGENTS######

actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_step_counter = tf.Variable(0)
global_step = tf.Variable(0,trainable=False)

starter_epsilon = .15
end_epsilon = .2
decay_steps = num_iterations + initial_collect_steps
ou_noise_size = tf.compat.v1.train.polynomial_decay(starter_epsilon,
                                                    train_step_counter,
                                                    decay_steps,
                                                    end_epsilon,
                                                    power=1.0,
                                                    cycle=False)
train_step = train_utils.create_train_step()


agent = ddpg_agent.DdpgAgent(
    time_step_spec= time_step_spec,
    action_spec= action_spec,
    actor_network= actor_net,
    critic_network= critic_net,
    actor_optimizer= actor_optimizer,
    critic_optimizer= critic_optimizer,
    ou_stddev= .2,
    ou_damping= 0.05,
    target_critic_network= None,
    target_update_tau= 0.001,
    dqda_clipping= None,
    td_errors_loss_fn= common.element_wise_squared_loss,
    gamma= 0.99,
    reward_scale_factor= None,
    gradient_clipping= True,
    debug_summaries= True,
    summarize_grads_and_vars= True,
    train_step_counter= train_step_counter,
    name= "Agent"
)


agent.initialize()

####REPLAY BUFFER####

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size= train_env.batch_size,
    max_length=replay_buffer_max_length,)
    # dataset_drop_remainder=True)


########POLICY#######
#Will be used to Populate Replay Buffer
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())
# compute_avg_return(eval_env, random_policy, num_eval_episodes)

#Defines Policies and run in Eager Mode, so computation is done immediately. Needed to prevent random "None" appearing
eval_policy = agent.policy
# eval_policy = eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
#   tf_eval_policy, use_tf_function=True)

collect_policy = agent.collect_policy
# collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
#   tf_collect_policy, use_tf_function=True)



#####METRICS####

average_return = tf_metrics.AverageReturnMetric(
    name='AverageReturn', 
    prefix='Metrics', 
    dtype=tf.float32, 
    batch_size=1,
    buffer_size=10
)

max_return = tf_metrics.MaxReturnMetric(
    name='MaxReturn', 
    prefix='Metrics', 
    dtype=tf.float32, 
    batch_size=1,
    buffer_size=10
)




####DRIVERS#####
observers = [average_return, max_return, replay_buffer.add_batch]
metric_observer = [average_return, max_return]

initial_collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    train_env,
    random_policy,
    observers=observers,
    num_episodes = 2)

collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    train_env,
    collect_policy,
    observers=observers,
    num_episodes = 1)

eval_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    eval_env,
    eval_policy,
    observers=observers,
    num_episodes = 1)

initial_collect_driver.run()

###REPLAY BUFFER###
dataset = replay_buffer.as_dataset(
    sample_batch_size = batch_size,
    num_steps = 2,
    single_deterministic_pass=False).prefetch(20)

experience_dataset_fn = lambda: dataset


#####DATA COLLECTION ####
def get_eval_metrics():
  eval_driver.run()
  results = {}
  for metric in metric_observer:
    results[metric.name] = metric.result()
  return results

metrics = get_eval_metrics()

def log_eval_metrics(step, metrics):
  eval_results = (', ').join(
      '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
  print('step = {0}: {1}'.format(step, eval_results))

log_eval_metrics(0, metrics)

####TRAIN####
# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
returns = []

iterator = iter(dataset)

for _ in range(num_iterations):
  # Training.
  collect_driver.run()
  for i in range(1000):
      trajectories, _ = next(iterator)
      loss_info = agent.train(experience=trajectories)
  # Note, step goes up by 1000 because of loop!
  step = agent.train_step_counter.numpy()
  if eval_interval and step % eval_interval == 0:
    metrics = get_eval_metrics()
    log_eval_metrics(step, metrics)
    returns.append(metrics["AverageReturn"])
####What is Loss info and why is it so different from Average Return?
  if log_interval and step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))



####PLOT#####
    
iterations = range(0, step, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=250)




