import numpy as np
import time
import sys
from acme.tf import utils as tf2_utils
from torch.nn import functional as F
import torch

def evaluate_tf(environment, actor, reps=10):
    
    episode_return = 0

    for i in range(reps):
        timestep = environment.reset()
        while not timestep.last():

            action = actor.select_action(timestep.observation)
            timestep = environment.step(action)

            # Have the agent observe the timestep and let the actor update itself.
            actor.observe(action, next_timestep=timestep)

            episode_return += timestep.reward
    return episode_return/reps

def evaluate_pytorch(environment, pyt_actor, reps=10):
    
    def input_from_obs(observation):
        observation = tf2_utils.add_batch_dim(observation)
        observation = tf2_utils.batch_concat(observation)
        return tf2_utils.to_numpy(observation)


    def pytorch_select_action(m, observation):
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      observation = input_from_obs(observation)
      observation = torch.from_numpy(observation).reshape(1, -1).to(device)
      #print(type(m[0]))
      #print(m[0].qmethod.in_features)
      #if m[0].method == "cutlass" and m[0].W_bits <= 8:
      #observation = F.pad(input=observation,
      #                    pad=(0, m[0].qmethod.in_features-observation.shape[1]),
      #                    mode="constant", value=0)
      #pass
      pred = m(observation)
      pred = pred.to("cpu")
      return pred.flatten().detach().numpy()
    
    episode_return = 0

    for i in range(reps):
        rew = 0
        timestep = environment.reset()
        while not timestep.last():
            action = pytorch_select_action(pyt_actor, timestep.observation)
            timestep = environment.step(action)
            rew += timestep.reward
        print("Attained: %f" % rew)
        sys.stdout.flush()
        episode_return += rew
    return episode_return/reps

def evaluate_pytorch_speed(environment, model, reps=10):
    def input_from_obs(observation):
        observation = tf2_utils.add_batch_dim(observation)
        observation = tf2_utils.batch_concat(observation)
        return tf2_utils.to_numpy(observation)

    speeds = []
    timestep = environment.reset()
    input = input_from_obs(timestep.observation)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.from_numpy(input).reshape(1, -1).to(device)

    for i in range(10):
        torch.cuda.synchronize()
        t_start = time.time()
        for j in range(100):
            model(input)
        torch.cuda.synchronize()
        t = time.time()-t_start
        speeds.append(t)

    return min(speeds)
            
