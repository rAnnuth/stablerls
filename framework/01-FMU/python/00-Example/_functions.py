#from gym import spaces
import types
import numpy as np
from gym import spaces 

def getActionSpace(self):
    low = np.array([-10]).astype(np.float32)
    high = np.array([10]).astype(np.float32)
    return spaces.Box(low, high)
#return spaces.discrete.Discrete(15, start=-7)

#TODO use attribute instead
def getObservationSpace(self):
    low = np.array([-5]).astype(np.float32)
    high = np.array([15]).astype(np.float32)
    self.isDiscrete = False # np.int64 for discrete or np.float32 for cont.
#    self.obsType = np.int64 # np.int64 for discrete or np.float32 for cont.
    return spaces.Box(low, high)
#return spaces.discrete.Discrete(30, start=-7)

#def resetIO(self):
    #pass
##return self.inputs = np.zeros([self.stopTime/self.dt, len(self.fmu.getInput)])


def getReward(self, action, observation):
    #self.input self.output also available
    # self.getMetric()
    if np.equal(observation,10):
        reward = 10
        done = True

    else:
        reward = -1 * float(abs(observation-5))
        done = False
    return observation, done, reward

def getMetric():
    pass

def _assignAction(self, action):
    # assign actions to inputs
    self.fmu.fmu.setReal([self.fmu.input[0].valueReference], [action])

def render(self, mode='human', close=False):
    #render to screen
    pass

def additionalStepInput(self):
    #use self.time to access simulation time and change input here
    self.fmu.fmu.setReal([self.fmu.input[1].valueReference], [5])
    #store in self.Inputs here
    pass

def exportResults(self):
    pass

