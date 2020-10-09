# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 14:35:56 2020

@author: Percy
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import random_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.specs import array_spec
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils
from tf_agents.agents.ddpg import critic_network
tf.compat.v1.enable_v2_behavior()

class ActorNetwork(network.Network):
  def __init__(self,
               observation_spec,
               action_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=[(32,3,3),(32,3,3),(32,3,3)],
               fc_layer_params=(200, 200),
               dropout_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               enable_last_layer_zero_initializer=False,
               name='ActorNetwork'):
    super(ActorNetwork, self).__init__(
        input_tensor_spec=observation_spec, state_spec=(), name=name)

    # For simplicity we will only support a single action float output.
    self._action_spec = action_spec
    flat_action_spec = tf.nest.flatten(action_spec)
##    if len(flat_action_spec) > 1:
##      raise ValueError('Only a single action is supported by this network')
#    self._single_action_spec = flat_action_spec[0]
#    if self._single_action_spec.dtype not in [tf.float32, tf.float64]:
#      raise ValueError('Only float actions are supported by this network.')

    kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1. / 3., mode='fan_in', distribution='uniform')
    self._encoder = encoding_network.EncodingNetwork(
        observation_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
        dropout_layer_params=dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        batch_squash=False)

    initializer = tf.keras.initializers.RandomUniform(
        minval=-0.0003, maxval=0.0003)

    self._action_projection_layer = tf.keras.layers.Dense(
        flat_action_spec[0].shape.num_elements(),
        activation=tf.keras.activations.tanh,
        kernel_initializer=initializer,
        name='action')

  def call(self, observations, step_type=(), network_state=()):
    outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
    # We use batch_squash here in case the observations have a time sequence
    # compoment.
    batch_squash = utils.BatchSquash(outer_rank)
    observations = tf.nest.map_structure(batch_squash.flatten, observations)

    state, network_state = self._encoder(
        observations, step_type=step_type, network_state=network_state)
    actions = self._action_projection_layer(state)
    actions = common_utils.scale_to_spec(actions, self._action_spec)
    actions = batch_squash.unflatten(actions)
    return tf.nest.pack_sequence_as(self._action_spec, [actions]), network_state





####ACTOR TEST####
action_spec = array_spec.BoundedArraySpec((6,), np.float32, minimum=0, maximum=10)
observation_spec = array_spec.BoundedArraySpec((64, 64, 3), np.float32, minimum=0,
                                        maximum=255)

random_env = random_py_environment.RandomPyEnvironment(observation_spec, action_spec=action_spec)

# Convert the environment to a TFEnv to generate tensors.
tf_env = tf_py_environment.TFPyEnvironment(random_env)

#preprocessing_layers = {
#    'image': tf.keras.models.Sequential([tf.keras.layers.Conv2D(8, 4),
#                                        tf.keras.layers.Flatten()]),
#    'vector': tf.keras.layers.Dense(5)
#    }
#preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
actor = ActorNetwork(tf_env.observation_spec(), 
                     tf_env.action_spec())

time_step = tf_env.reset()
print(actor(time_step.observation,time_step.step_type))