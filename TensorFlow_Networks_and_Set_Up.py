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
from tf_agents.utils import common
from tf_agents.policies import py_tf_eager_policy
from Networks import ActorNetwork
tf.compat.v1.enable_v2_behavior()


tempdir = tempfile.gettempdir()

###################HYPERPARAMETERS##################
num_iterations = 20000 # @param {type:"integer"}

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
actor_net = ActorNetwork(observation_spec,
               action_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params= conv_layer_params,
               fc_layer_params=fc_layer_params,
               dropout_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               enable_last_layer_zero_initializer=False,
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
    actor_optimizer=actor_optimizer,
    critic_optimizer= critic_optimizer,
    ou_stddev= .1,
    ou_damping= 0.25,
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
tf_eval_policy = agent.policy
eval_policy = eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
  tf_eval_policy, use_tf_function=True)

tf_collect_policy = agent.collect_policy
collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
  tf_collect_policy, use_tf_function=True)


#####ACTORS######
#Fills out Replay Buffer with Random Policy (NOTE: May require some function or something as an observer)
# replay_observer = [replay_buffer]
# transition_observer = trajectory.from_transition
# print(train_env)
# print(random_policy)
# print(train_step)
# print(initial_collect_steps)
# print(replay_observer)
# initial_collect_actor = actor.Actor(
#   train_env,
#   random_policy,
#   train_step,
#   steps_per_run=initial_collect_steps,
#   observers=[replay_observer])
# initial_collect_actor.run()
# #Used to gather experience during training
# env_step_metric = py_metrics.EnvironmentSteps()
# collect_actor = actor.Actor(
#   train_py_env,
#   collect_policy,
#   train_step,
#   steps_per_run=1,
#   metrics=actor.collect_metrics(10),
#   summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
#   observers=[replay_observer, env_step_metric])

# #Used to evaluate policies during training
# eval_actor = actor.Actor(
#   eval_env,
#   eval_policy,
#   train_step,
#   episodes_per_run=num_eval_episodes,
#   metrics=actor.eval_metrics(num_eval_episodes),
#   summary_dir=os.path.join(tempdir, 'eval'),
# )

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
# print(average_return.result().numpy())
dataset = replay_buffer.as_dataset(
    sample_batch_size = batch_size,
    num_steps = 2,
    single_deterministic_pass=False).prefetch(20)

experience_dataset_fn = lambda: dataset

####LEARNER####
# saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

# Triggers to save the agent's policy checkpoints.
# learning_triggers = [
#     triggers.PolicySavedModelTrigger(
#         saved_model_dir,
#         agent,
#         train_step,
#         interval=policy_save_interval),
#     triggers.StepPerSecondLogTrigger(train_step, interval=1000),
# ]

# agent_learner = learner.Learner(
#   tempdir,
#   train_step,
#   agent,
#   experience_dataset_fn,
#   triggers=learning_triggers)



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
avg_return = get_eval_metrics()["AverageReturn"]
returns = [avg_return]

iterator = iter(dataset)

for _ in range(num_iterations):
  # Training.
  collect_driver.run()
  for i in range(1000):
      trajectories, _ = next(iterator)
      loss_info = agent.train(experience=trajectories)
  # Note, step goes up by 500 because of loop!
  step = agent.train_step_counter.numpy()
  if eval_interval and step % eval_interval == 0:
    metrics = get_eval_metrics()
    log_eval_metrics(step, metrics)
    returns.append(metrics["AverageReturn"])
####What is Loss info and why is it so different from Average Return?
  if log_interval and step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))




# print("we got to 2")
# def compute_avg_return(environment, policy, num_episodes=10):

#   total_return = 0.0
#   for _ in range(num_episodes):

#     time_step = environment.reset()
#     episode_return = 0.0

#     while not time_step.is_last():
#       action_step = policy.action(time_step)
#       time_step = environment.step(action_step.action)
#       episode_return += time_step.reward
# #      print(time_step.reward)
#     total_return += episode_return

#   avg_return = total_return / num_episodes
#   return avg_return.numpy()[0]

# #compute_avg_return(eval_env, eval_policy, num_eval_episodes)


# def collect_step(environment, policy, buffer):
#   time_step = environment.current_time_step()
#   action_step = policy.action(time_step)
#   next_time_step = environment.step(action_step.action)
#   traj = trajectory.from_transition(time_step, action_step, next_time_step)
#   # Add trajectory to the replay buffer
#   buffer.add_batch(traj)

# def collect_data(env, policy, buffer, steps):
#   for _ in range(steps):
#     collect_step(env, policy, buffer)
# print("we got to 3")
#collect_data(train_env, collect_policy, replay_buffer, steps=100)

# This loop is so common in RL, that we provide standard implementations. 
# For more details see the drivers module.
# https://www.tensorflow.org/agents/api_docs/python/tf_agents/drivers




#####TRAIN AND EXECUTE #####
#try:
#  %%time
#except:
#  pass

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
#agent.train = common.function(agent.train)

# Reset the train step
#agent.train_step_counter.assign(0)

# Peform functions required before entering the training loop
###Observer###


# collect_steps_per_iteration = 64
# collect_op = dynamic_step_driver.DynamicStepDriver(
#   train_env,
#   agent.collect_policy,
#   observers=replay_observer,
#   num_steps=collect_steps_per_iteration).run()



# iterator = iter(dataset)
# # collect_data(train_env, random_policy, replay_buffer, steps=100)
# # agent.train = common.function(agent.train)
# agent.train_step_counter.assign(0)
# avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
# returns = [avg_return]
# # print(returns)

# # Evaluate the agent's policy once before training.
# # print("we got to 4")
# for _ in range(num_iterations):

#   # Collect a few steps using collect_policy and save to the replay buffer.
#   for _ in range(collect_steps_per_iteration):
#     collect_step(train_env, agent.collect_policy, replay_buffer)

#   # Sample a batch of data from the buffer and update the agent's network.
#   experience, unused_info = next(iterator)
#   train_loss = agent.train(experience).loss
#   # print("Agent Trained!")
#   step = agent.train_step_counter.numpy()

#   if step % log_interval == 0:
#     print('step = {0}: loss = {1}'.format(step, train_loss))

#   if step % eval_interval == 0:
#     avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
#     print('step = {0}: Average Return = {1}'.format(step, avg_return))
#     returns.append(avg_return)


####PLOT#####
    
iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=250)




