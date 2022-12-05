#from gym import spaces
import types
import numpy as np
from gym import spaces 

#np.seterr(all='raise') 

def getActionSpace(self):
    low = np.array([-2e3]).astype(np.float32)
    high = np.array([2e3]).astype(np.float32)
    return spaces.Box(low, high)
#return spaces.discrete.Discrete(15, start=-7)

#TODO use attribute instead
def getObservationSpace(self):
    low = np.array([-np.inf, -np.inf]).astype(np.float32)
    high = np.array([np.inf, np.inf]).astype(np.float32)
    #self.isDiscrete = False # np.int64 for discrete or np.float32 for cont.
#   self.obsType = np.int64 # np.int64 for discrete or np.float32 for cont.
    return spaces.Box(low, high)
#return spaces.discrete.Discrete(30, start=-7)

#def resetIO(self):
    #pass
##return self.inputs = np.zeros([self.stopTime/self.dt, len(self.fmu.getInput)])

def _reset(self):

    self.fmu.fmu.setReal([self.fmu.input[0].valueReference], [0])
    self.fmu.fmu.setReal([self.fmu.input[1].valueReference], [self.Href+np.random.uniform(low=-0.5, high=0.5)])
    #self.fmu.fmu.setReal([self.fmu.input[1].valueReference], [self.Href])
    self._nextObservation(1)
    return self.outputs[self.stepCount,:][1:]


def getReward(self, action, observation):
    #self.input self.output also available
    # self.getMetric()
    if (self.outputs[-1,0] > 1e3):
      print('Strange height')
      print(self.outputs[-1,:])

    #error = self.outputs[:,0].astype(np.float64)
    #integral = self.outputs[:,1]
    #uctrls = self.inputs[:,0].astype(np.float64)
    observation = observation[1:]
    prevAction = self.inputs[self.stepCount - 1, 0] #TODO usign action and not prev
    #reward = -(observation[0]**2 + 0.01 * action**2)[0]
    reward = -(observation[0]**2 + 0.01 * prevAction**2) #TODO 0.05 instead 0.01
    done = False

    if np.isinf(observation).any() or np.isnan(observation).any():
        #minVal = np.finfo(np.float32).min/1e30
        #observation = np.array([minVal, minVal])
        print('inf observation')
        import sys
        sys.exit()
    return observation, reward, done

def getMetric():
    pass

#def obsProcessing(self, observation):
    #return observation 

def _assignAction(self, action):
    # assign actions to inputs
    self.fmu.fmu.setReal([self.fmu.input[0].valueReference], [action[0]])

def render(self, mode='human', close=False):
    #render to screen
    pass

def FMUstep(self):
    #use self.time to access simulation time and change input here
    #store in self.Inputs here
    #np.nditer() solves problems!
    pass


def exportResults(self):
    pass

