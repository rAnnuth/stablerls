#from gym import spaces
import types
import numpy as np
from gym import spaces 

def getActionSpace(self):
    low = np.array([0, 0]).astype(np.float32)
    high = np.array([20, 2]).astype(np.float32)
    return spaces.Box(low, high)
#return spaces.discrete.Discrete(15, start=-7)

#TODO use attribute instead
def getObservationSpace(self):
    low = np.array([-100]).astype(np.float32)
    high = np.array([100]).astype(np.float32)
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
    
    reward = -((self.Href-observation[0])**2 + 0.01*observation[1]**2)
    observation = self.Href - observation[0]

    done = False

    #self.errorSum += observation[0]

    return done, reward

def getMetric():
    pass

def obsProcessing(self, observation):
    observation = np.array(self.Href - observation[0])
    obs = list()
    obs.append(observation.item())

    return obs 

def _assignAction(self, action):
    # assign actions to inputs

    self.fmu.fmu.setReal([self.fmu.input[0].valueReference], [1e3*action[0]])
    self.fmu.fmu.setReal([self.fmu.input[1].valueReference], [action[1]])

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

