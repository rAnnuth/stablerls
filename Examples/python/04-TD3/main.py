# DEBUG = os.getenv('DEBUG', False) is not None
# LOG = os.getenv('LOG', False) is not None

import os
import fmuSimulation.gymFMU 
import numpy as np
import sys

config = os.path.abspath('Example.cfg')

env = fmuSimulation.gymFMU.gymFMU(config)
obs = env.reset()

action = np.array([10])

done = False
for i in np.arange(10):
    #    print(i)
    observation, reward, done, _ = env.step(action)
    #print(observation, reward, done)
        
#env.exportResults()
#env.plotResults()
env.close()

## Config

#read fmu
#TODO auto update

#io definition
#action definition
#algorithm definition:
