import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import time
import pickle
import acme
from acme import adders
from acme import specs
from acme import core
from acme import types
from acme import wrappers
from acme.agents.tf import actors
from acme.tf import savers as tf2_savers
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from collections import OrderedDict
from concurrent import futures
import gym
from typing import Mapping, Optional, Sequence
from typing import Mapping, Sequence
import msgpack
import numpy as np
import random
import sonnet as snt
import sys
from dm_control import suite
import tensorflow as tf
import tensorflow_probability as tfp
import time
import torch
import traceback
import tree
import os
import loop

from qlinear import QLinear

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="gym,MountainCarContinuous-v0", type=str)
parser.add_argument("--model_path", default=None, required=True, type=str)
parser.add_argument("--model_str", default="4096,4096,4096", type=str)
parser.add_argument("--method", type=str, default="cutlass", choices=["cutlass","pbatch","fake"])

parser.add_argument("--W_bits_1", type=int, default=32)
parser.add_argument("--A_bits_1", type=int, default=32)

parser.add_argument("--W_bits_2", type=int, default=32)
parser.add_argument("--A_bits_2", type=int, default=32)

parser.add_argument("--W_bits_3", type=int, default=32)
parser.add_argument("--A_bits_3", type=int, default=32)

#parser.add_argument("--pbatch_act_shift", type=int, default=2**6)
#parser.add_argument("--pbatch_act_shift", type=int, default=900)
#parser.add_argument("--pbatch_act_shift", type=int, default=750)
parser.add_argument("--pbatch_act_shift", type=int, default=200)
parser.add_argument("--optimize_cutoff", type=int, default=1)

class PytorchTanhToSpec(torch.nn.Module):
  def __init__(self, action_spec):
    super(PytorchTanhToSpec, self).__init__()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self._scale = torch.from_numpy(np.array(action_spec.maximum - action_spec.minimum)).to(device)
    self._offset = torch.from_numpy(np.array(action_spec.minimum)).to(device)

  def forward(self, x):
    inputs = torch.tanh(x)
    inputs = .5 * (inputs + 1.0)
    outputs = inputs * self._scale + self._offset
    return outputs

class PytorchClippedGaussian(torch.nn.Module):
  def __init__(self, sigma):
    super(PytorchClippedGaussian, self).__init__()
    self.sigma = sigma

  def forward(self, x):
    return x

class PytorchClipToSpec(torch.nn.Module):
  def __init__(self, spec):
    super(PytorchClipToSpec, self).__init__()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.spec = spec
    self._min = torch.from_numpy(spec.minimum).to(device)
    self._max = torch.from_numpy(spec.maximum).to(device)

  def forward(self, x):
    return torch.max(torch.min(x, self._max), self._min)

def input_size_from_obs_spec(env_spec):
    if hasattr(env_spec, "shape"):
        return int(np.prod(env_spec.shape))
    if type(env_spec) == OrderedDict:
        return int(sum([input_size_from_obs_spec(x) for x in env_spec.values()]))
    try:
        return int(sum([input_size_from_obs_spec(x) for x in env_spec]))
    except:
        assert(0)

def create_model(input_size, output_size, action_spec, args, policy_layer_sizes=(2048,2048,2048)):
    # Create policy network
    # Pytorch equivalent of: https://github.com/deepmind/acme/blob/master/acme/tf/networks/continuous.py
    # First layer
    sizes = [input_size] + list(policy_layer_sizes) + [output_size]
    layers = []
    for i in range(len(sizes)-1):
        in_size, out_size = sizes[i], sizes[i+1]
        if i != 0:
          bits = {
            1: (args.W_bits_1, args.A_bits_1),
            2: (args.W_bits_2, args.A_bits_2),
            3: (args.W_bits_3, args.A_bits_3),
          }
          W_bits, A_bits = bits[i]
          layers.append(QLinear(in_size, out_size,
                                W_bits=W_bits, A_bits=A_bits,
                                method=args.method, optimize_cutoff=args.optimize_cutoff,
                                pbatch_act_shift=args.pbatch_act_shift))
        else:
          layers.append(QLinear(in_size, out_size,
                                W_bits=32, A_bits=32,
                                method="fake",
                                optimize_cutoff=args.optimize_cutoff))
          #layers.append(torch.nn.Linear(in_size, out_size))
          
        layers.append(torch.nn.ReLU())

    layers = layers[:-1]
    layers.append(PytorchTanhToSpec(action_spec))
    layers.append(PytorchClipToSpec(action_spec))
    return torch.nn.Sequential(*layers)

def pytorch_model_load_state_dict(model, new_variables):
    if len(new_variables) == 0:
      return

    pytorch_keys = list(model.state_dict().keys())
    pytorch_keys = [x for x in pytorch_keys if "qmethod" not in x]
    # Switch ordering of weight + bias
    new_pytorch_keys = []
    for i in range(0, len(pytorch_keys), 2):
      new_pytorch_keys.append(pytorch_keys[i+1])
      new_pytorch_keys.append(pytorch_keys[i])
    pytorch_keys = new_pytorch_keys
    new_state_dict = {k:torch.from_numpy(v.T) for k,v in zip(pytorch_keys, new_variables)}
    model.load_state_dict(new_state_dict)

def make_environment(
        taskstr="gym,MountainCarContinuous-v0"):
  
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
        activation=tf.nn.tanh,
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

def evaluate_tf(args):
    model_tuple = tuple([int(x) for x in args.model_str.split(",")])
    environment = make_environment(taskstr=args.env)
    environment_spec = specs.make_environment_spec(environment)
    agent_networks = make_networks(environment_spec.actions, policy_layer_sizes=model_tuple)

    obs_net = tf2_utils.to_sonnet_module(tf2_utils.batch_concat)
    emb_spec = tf2_utils.create_variables(obs_net, [environment_spec.observations])
    tf2_utils.create_variables(agent_networks["policy"], [emb_spec])

    # Create the evaluation policy.
    eval_policy = snt.Sequential([
        agent_networks['observation'],
        agent_networks['policy'],
    ])

    with open(args.model_path, "rb") as f:
        weights = pickle.load(f)
    for tf_v, v in zip(agent_networks["policy"].variables, weights):
        tf_v.assign(v)
    eval_actor = actors.FeedForwardActor(policy_network=eval_policy)
    return loop.evaluate_tf(environment, eval_actor)

def evaluate_pytorch(args):
    model_tuple = tuple([int(x) for x in args.model_str.split(",")])
    environment = make_environment(taskstr=args.env)
    environment_spec = specs.make_environment_spec(environment)

    with open(args.model_path, "rb") as f:
        weights = pickle.load(f)
    
    pytorch_model = create_model(
        input_size_from_obs_spec(environment_spec.observations),
        np.prod(environment_spec.actions.shape, dtype=int),
        environment_spec.actions,
        args=args,
        policy_layer_sizes=model_tuple)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pytorch_model.to(device)
    
    pytorch_model_load_state_dict(pytorch_model, weights)
    return loop.evaluate_pytorch(environment, pytorch_model), loop.evaluate_pytorch_speed(environment, pytorch_model)
        
if __name__=="__main__":
    args = parser.parse_args()
    #print(evaluate_tf(args))    
    quality, time = evaluate_pytorch(args)
    env, method = args.env, args.method,
    W1,W2,W3 = args.W_bits_1,args.W_bits_2,args.W_bits_3
    A1,A2,A3 = args.A_bits_1,args.A_bits_2,args.A_bits_3
    print("%s,%s,%d,%d,%d,%d,%d,%d,%f,%f" % (
      env,method, W1,A1, W2,A2, W3,A3, quality, time
    ))
