import gym, ray
from ray.rllib.algorithms import a3c

from torchModel import customTorchModel, FullyConnectedNetwork


import fmuSimulation.gymFMU as ExampleFMU
from fmuSimulation.configReader import configReader
import os
config = os.path.abspath('Example.cfg')
cfg = configReader(config)
config = cfg.getAgent()
config['framework'] = 'torch'
config['model'] = {}
config['model']['custom_model'] = FullyConnectedNetwork

agent = a3c.A3C(env=ExampleFMU.gymFMU, config=config)

import pdb
p = agent.get_policy() 
pdb.set_trace()
a = p.compute_single_action([1,1])
print(a)
