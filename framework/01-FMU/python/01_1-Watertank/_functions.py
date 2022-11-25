#from gym import spaces
import types
import numpy as np
from gym import spaces 

#np.seterr(all='raise') 

def getActionSpace(self):
    low = np.array([0,0]).astype(np.float32)
    high = np.array([30,30]).astype(np.float32)
    return spaces.Box(low, high)
#return spaces.discrete.Discrete(15, start=-7)

#TODO use attribute instead
def getObservationSpace(self):
    low = np.array([-np.inf, -np.inf, -np.inf]).astype(np.float32)
    high = np.array([np.inf, np.inf, np.inf]).astype(np.float32)
    #self.isDiscrete = False # np.int64 for discrete or np.float32 for cont.
#   self.obsType = np.int64 # np.int64 for discrete or np.float32 for cont.
    return spaces.Box(low, high)
#return spaces.discrete.Discrete(30, start=-7)

#def resetIO(self):
    #pass
##return self.inputs = np.zeros([self.stopTime/self.dt, len(self.fmu.getInput)])

def _reset(self):
    self.fmu.fmu.setReal([self.fmu.input[0].valueReference], [2])
    self.fmu.fmu.setReal([self.fmu.input[1].valueReference], [0])
    self.fmu.fmu.setReal([self.fmu.input[2].valueReference], [0])
    self.fmu.fmu.setReal([self.fmu.input[3].valueReference], [10])
    self._nextObservation(1)
    
    return self.outputs[self.stepCount,:] 


def getReward(self, action, observation):
    #self.input self.output also available
    # self.getMetric()

    error = np.sum(np.abs(self.outputs[:,1]))
    #integral = np.sum(np.abs(self.outputs[:,1]))
    uctrls = np.sum(np.abs(self.inputs[:,0]))
    #reward = -(np.abs(observation[1])**2 + 0.01 * np.abs(observation[0])**2)
    reward = -((error)**2 + 0.01 * (uctrls)**2)

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

#def obsProcessing(self, observation):
    #return observation 

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
        print(f'using {u}')
    else:
        u = [0]
        print(f'using {u}')
    self.u.append(u)
    self.fmu.fmu.setReal([self.fmu.input[2].valueReference], u)

    


def exportResults(self):
    pass

