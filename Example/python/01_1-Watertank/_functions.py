#from gym import spaces
import types
import numpy as np
from gym import spaces 

#np.seterr(all='raise') 

def getActionSpace(self):
    low = np.array([0,0]).astype(np.float32)
    high = np.array([40,40]).astype(np.float32)
    return spaces.Box(low, high)
#return spaces.discrete.Discrete(15, start=-7)

#TODO use attribute instead
def getObservationSpace(self):
    low = np.full([1,1001], -np.inf, dtype=np.float32)
    high = np.full([1,1001], np.inf, dtype=np.float32)
    #low = np.array([-np.inf, -np.inf, -np.inf]).astype(np.float32)
    #high = np.array([np.inf, np.inf, np.inf]).astype(np.float32)
    #self.isDiscrete = False # np.int64 for discrete or np.float32 for cont.
#   self.obsType = np.int64 # np.int64 for discrete or np.float32 for cont.
    return spaces.Box(low, high)
#return spaces.discrete.Discrete(30, start=-7)

#def resetIO(self):
    #pass
##return self.inputs = np.zeros([self.stopTime/self.dt, len(self.fmu.getInput)])

def _reset(self):
    self.fmu.fmu.setReal([self.fmu.input[0].valueReference], [0])
    self.fmu.fmu.setReal([self.fmu.input[1].valueReference], [0])
    self.fmu.fmu.setReal([self.fmu.input[2].valueReference], [0]) 
    self.fmu.fmu.setReal([self.fmu.input[3].valueReference], [self.Href + np.random.randn()*5])
    self.customVars['ctrlInt'] = np.array([0]).astype(np.float32)
    self._nextObservation(1)
    
    return np.full([1,1001], 0, dtype=np.float32)


def getReward(self, action, observation):
    #self.input self.output also available
    # self.getMetric()

    error = np.sum((self.outputs[:,1])**2) * self.dt
    #integral = np.sum(np.abs(self.outputs[:,1]))
    #uctrls = np.sum(np.abs(self.inputs[:,0]))
    #reward = -(np.abs(observation[1])**2 + 0.01 * np.abs(observation[0])**2)
    reward = -((error) + 0.01 *self.customVars['ctrlInt'])[0]
    done = False
    if action.any() < 0:
        reward += -1e4
        print('negative action')
        done = True

    if np.isinf(observation).any() or np.isnan(observation).any():
        #minVal = np.finfo(np.float32).min/1e30
        #observation = np.array([minVal, minVal])
        reward += -1e4
        print('inf observation')
        done = True
        
    return observation, reward, done

def getMetric():
    pass

def obsProcessing(self, observation):
    observation = self.outputs[:,0].reshape([1,1001])
    return observation 

def _assignAction(self, action):
    # assign actions to inputs
    self.fmu.fmu.setReal([self.fmu.input[0].valueReference], [action[0]])
    self.fmu.fmu.setReal([self.fmu.input[1].valueReference], [action[1]])

def render(self, mode='human', close=False):
    #render to screen
    pass

def FMUstep(self):
    #use self.time to access simulation time and change input here
    #store in self.Inputs here
    #np.nditer() solves problems!
    if self.stepCount >= 0:
        Ki, Kp = np.array([self.fmu.fmu.getReal([x.valueReference])[0] for x in self.fmu.getInput()])[[0,1]]
        error , errorInt = np.array([self.fmu.fmu.getReal([x.valueReference])[0] for x in self.fmu.getOutput()])[1:]
        u = np.array([Ki* errorInt + Kp*error])
    
    else:
        error , errorInt = np.array([self.fmu.fmu.getReal([x.valueReference])[0] for x in self.fmu.getOutput()])[1:]
        u = np.array([0])

    self.customVars['ctrlInt'] += (self.dt * u**2).astype(np.float32)
    self.fmu.fmu.setReal([self.fmu.input[2].valueReference], u)

    


def exportResults(self):
    pass

