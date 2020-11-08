# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 18:02:38 2020

@author: Percy
"""

import base64
import imageio
import IPython
import matplotlib.pyplot as plt
import os
import reverb
import tempfile
import PIL.Image

import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_pybullet
from tf_agents.experimental.train import actor
from tf_agents.experimental.train import learner
from tf_agents.experimental.train import triggers
from tf_agents.experimental.train.utils import spec_utils
from tf_agents.experimental.train.utils import strategy_utils
from tf_agents.experimental.train.utils import train_utils
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.replay_buffers import table
tempdir = tempfile.gettempdir()

env_name = "MinitaurBulletEnv-v0" # @param {type:"string"}

# Use "num_iterations = 1e6" for better results (2 hrs)
# 1e5 is just so this doesn't take too long (1 hr)
num_iterations = 100000 # @param {type:"integer"}

initial_collect_steps = 10000 # @param {type:"integer"}
collect_steps_per_iteration = 1 # @param {type:"integer"}
replay_buffer_capacity = 10000 # @param {type:"integer"}

batch_size = 256 # @param {type:"integer"}

critic_learning_rate = 3e-4 # @param {type:"number"}
actor_learning_rate = 3e-4 # @param {type:"number"}
alpha_learning_rate = 3e-4 # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

log_interval = 5000 # @param {type:"integer"}

num_eval_episodes = 20 # @param {type:"integer"}
eval_interval = 10000 # @param {type:"integer"}

policy_save_interval = 5000 # @param {type:"integer"}


env = suite_pybullet.load(env_name)
env.reset()
PIL.Image.fromarray(env.render())

collect_env = suite_pybullet.load(env_name)
eval_env = suite_pybullet.load(env_name)

observation_spec, action_spec, time_step_spec = (
      spec_utils.get_tensor_specs(collect_env))

conv_layer_params = None
fc_layer_params=(200, 200)
kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1. / 3., mode='fan_in', distribution='uniform')
final_layer_initializer = tf.keras.initializers.RandomUniform(
        minval=-0.0003, maxval=0.0003)
actor_net = actor_distribution_network.ActorDistributionNetwork(
      observation_spec,
      action_spec,
      fc_layer_params=actor_fc_layer_params,
      continuous_projection_net=(
          tanh_normal_projection_network.TanhNormalProjectionNetwork))


critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_conv_layer_params=conv_layer_params,
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=fc_layer_params,
        kernel_initializer=kernel_initializer,
        last_kernel_initializer=final_layer_initializer)

target_actor_net = actor_distribution_network.ActorDistributionNetwork(
      observation_spec,
      action_spec,
      fc_layer_params=actor_fc_layer_params,
      continuous_projection_net=(
          tanh_normal_projection_network.TanhNormalProjectionNetwork))


target_critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_conv_layer_params=conv_layer_params,
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=fc_layer_params,
        kernel_initializer=kernel_initializer,
        last_kernel_initializer=final_layer_initializer)


actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_step = train_utils.create_train_step()

tf_agent = ddpg_agent.DdpgAgent(
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
    train_step_counter= train_step,
    name= "Agent"
)


#server = reverb.Server(tables=[
#    reverb.Table(
#        name='my_table',
#        sampler=reverb.selectors.Uniform(),
#        remover=reverb.selectors.Fifo(),
#        max_size=100,
#        rate_limiter=reverb.rate_limiters.MinSize(1)),
#    ],
#    port=8000
#)
table_name = 'uniform_table'
#table = table.Table(
#    tf_agent.collect_data_spec,
#    capacity=replay_buffer_capacity, 
#    scope=table_name)
##    
#reverb_server = reverb.Server([table])

#reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
#    tf_agent.collect_data_spec,
#    sequence_length=2,
#    table_name=table_name,
#    local_server = server)


dataset = replay_buffer.as_dataset(
      sample_batch_size=batch_size, num_steps=2).prefetch(50)
experience_dataset_fn = lambda: dataset

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy
random_policy = random_py_policy.RandomPyPolicy(
  collect_env.time_step_spec(), collect_env.action_spec())

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  reverb_replay.py_client,
  table_name,
  sequence_length=2,
  stride_length=1)

initial_collect_actor = actor.Actor(
  collect_env,
  random_policy,
  train_step,
  steps_per_run=initial_collect_steps,
  observers=[rb_observer])
initial_collect_actor.run()

env_step_metric = py_metrics.EnvironmentSteps()
collect_actor = actor.Actor(
  collect_env,
  collect_policy,
  train_step,
  steps_per_run=1,
  metrics=actor.collect_metrics(10),
  summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
  observers=[rb_observer, env_step_metric])

eval_actor = actor.Actor(
  eval_env,
  eval_policy,
  train_step,
  episodes_per_run=num_eval_episodes,
  metrics=actor.eval_metrics(num_eval_episodes),
  summary_dir=os.path.join(tempdir, 'eval'),
)

saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

# Triggers to save the agent's policy checkpoints.
learning_triggers = [
    triggers.PolicySavedModelTrigger(
        saved_model_dir,
        tf_agent,
        train_step,
        interval=policy_save_interval),
    triggers.StepPerSecondLogTrigger(train_step, interval=1000),
]

agent_learner = learner.Learner(
  tempdir,
  train_step,
  tf_agent,
  experience_dataset_fn,
  triggers=learning_triggers)

def get_eval_metrics():
  eval_actor.run()
  results = {}
  for metric in eval_actor.metrics:
    results[metric.name] = metric.result()
  return results

metrics = get_eval_metrics()


def log_eval_metrics(step, metrics):
  eval_results = (', ').join(
      '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
  print('step = {0}: {1}'.format(step, eval_results))

log_eval_metrics(0, metrics)

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = get_eval_metrics()["AverageReturn"]
returns = [avg_return]

for _ in range(num_iterations):
  # Training.
  collect_actor.run()
  loss_info = agent_learner.run(iterations=1)

  # Evaluating.
  step = agent_learner.train_step_numpy

  if eval_interval and step % eval_interval == 0:
    metrics = get_eval_metrics()
    log_eval_metrics(step, metrics)
    returns.append(metrics["AverageReturn"])

  if log_interval and step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

rb_observer.close()
reverb_server.stop()



steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim()

