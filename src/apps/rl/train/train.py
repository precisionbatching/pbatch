# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example running D4PG on the control suite."""
import sys
from typing import Mapping, Sequence
import pickle
from absl import app
from absl import flags
import acme
from acme import specs
from acme import types
from acme import wrappers
from acme.agents.tf import actors
#from acme.agents.tf import d4pg
from d4pg_rewrite import D4PG
from acme.tf import networks
from acme.tf import utils as tf2_utils
from dm_control import suite
from acme.tf import savers as tf2_savers
import dm_env
import numpy as np
import sonnet as snt
import gym
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "gym,MountainCarContinuous-v0", "env")
flags.DEFINE_string("model_str", "4096,4096,4096", "model_str")
flags.DEFINE_integer('num_episodes', 100,
                     'Number of training episodes to run for.')

flags.DEFINE_integer('num_episodes_per_eval', 10,
                     'Number of training episodes to run between evaluation '
                     'episodes.')
flags.DEFINE_integer("min_replay_size", 10000, "min_replay_size")
flags.DEFINE_integer("max_replay_size", 300000, "max_replay_size")
flags.DEFINE_float("sigma", .2, "sigma")

def make_environment(
        taskstr="gym,MountainCarContinuous-v0") -> dm_env.Environment:
  
  """Creates an OpenAI Gym environment."""

  # Load the gym environment.
  module, task = taskstr.split(",")

  if module == "gym":
    environment = gym.make(task)
    environment = wrappers.GymWrapper(environment)  
  elif module == "atari":
    environment = gym.make(task, full_action_space=True)
    environment = gym_wrapper.GymAtariAdapter(environment)
    environment = atari_wrapper.AtariWrapper(environment, to_float=True, max_episode_len=5000, zero_discount_on_life_loss=True,
)
  elif module == "dm_control":
    t1,t2 = task.split(":")
    environment = suite.load(t1, t2)
  elif module == "bsuite":
    environment = bsuite.load_and_record_to_csv(
      bsuite_id=task,
      results_dir="./bsuite_results"
    )
    

  # Make sure the environment obeys the dm_env.Environment interface.
  environment = wrappers.SinglePrecisionWrapper(environment)

  return environment

def make_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    vmin: float = -150.,
    vmax: float = 150.,
    num_atoms: int = 51,
    placement: str="CPU",
) -> Mapping[str, types.TensorTransformation]:
  """Creates the networks used by the agent."""

  # Get total number of action dimensions from action spec.
  num_dimensions = np.prod(action_spec.shape, dtype=int)    

  # Create the shared observation network; here simply a state-less operation.
  observation_network = tf2_utils.batch_concat

  uniform_initializer = tf.initializers.VarianceScaling(
    distribution='uniform', mode='fan_out', scale=0.333)
  network = snt.Sequential([
    snt.nets.MLP(
        policy_layer_sizes,
        w_init=uniform_initializer,
        activation=tf.nn.relu,
        activate_final=False
    )])

  # Create the policy network.
  policy_network = snt.Sequential([
    network,
    networks.NearZeroInitializedLinear(num_dimensions),
    networks.TanhToSpec(action_spec)])

  # Create the critic network.
  critic_network = snt.Sequential([
    # The multiplexer concatenates the observations/actions.
    networks.CriticMultiplexer(),
    networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
    networks.DiscreteValuedHead(vmin, vmax, num_atoms),
  ])

  return {
    'policy': policy_network,
    'critic': critic_network,
    'observation': observation_network,
  }

def save_model(env,policy):
  with open("%s_checkpoint.json" % env, "wb") as f:
    weights = [tf2_utils.to_numpy(v) for v in policy.variables]
    pickle.dump(weights, f)

def main(_):
  # Create an environment, grab the spec, and use it to create networks.
  model_tuple = tuple([int(x) for x in FLAGS.model_str.split(",")])
  environment = make_environment(taskstr=FLAGS.env)
  environment_spec = specs.make_environment_spec(environment)
  agent_networks = make_networks(environment_spec.actions, policy_layer_sizes=model_tuple)

  # Construct the agent.
  agent = D4PG(
      environment_spec=environment_spec,
      policy_network=agent_networks['policy'],
      critic_network=agent_networks['critic'],
      observation_network=agent_networks['observation'],  # pytype: disable=wrong-arg-types
      min_replay_size=FLAGS.min_replay_size,
      max_replay_size=FLAGS.max_replay_size,
      sigma=FLAGS.sigma,
      n_step=3,
      checkpoint=False,      
  )

  # Create the environment loop used for training.
  train_loop = acme.EnvironmentLoop(environment, agent, label='train_loop')

  # Create the evaluation policy.
  eval_policy = snt.Sequential([
      agent_networks['observation'],
      agent_networks['policy'],
  ])

  # Create the evaluation actor and loop.
  eval_actor = actors.FeedForwardActor(policy_network=eval_policy)
  eval_env = make_environment(taskstr=FLAGS.env)
  eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop')

  for _ in range(FLAGS.num_episodes // FLAGS.num_episodes_per_eval):
    train_loop.run(num_episodes=FLAGS.num_episodes_per_eval)
    eval_loop.run(num_episodes=1)
    print("Saving...")
    save_model(FLAGS.env, agent_networks["policy"])
    sys.stdout.flush()

if __name__ == '__main__':
  app.run(main)
