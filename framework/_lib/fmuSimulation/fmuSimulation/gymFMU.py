# -*- coding: utf-8 -*-

# name: cabinEnv.py
# description: class definition for a gym environment which uses a fmu to simulate a dc microgrid,
#               main can be used for testing the environment, microgridCabin class can be used for 
#               various control strategies but especially to train RL agents, environment contains
#               two definitions of actions (one is pure Discrete, the other MultiDiscrete)
# author: Finn Nußbaum (crw3068)
# version/date: V1.3, 27.09.2022

# ----------------------------------------------------------------------------
# external libraries
import types, contextlib, json
import gym

import numpy as np
from gym import spaces
from fmuSimulation.fmutools import FMU
from fmuSimulation.smartParse import smartParse
# ----------------------------------------------------------------------------
section_names = 'Reinforcement Learning', 'General', 'FMU'
# ----------------------------------------------------------------------------
class gymFMU(gym.Env):
    '''
    Custom environment for simulation of FMUs with gym interface
    See https://www.gymlibrary.dev/ for information about API and necessary functions.
    '''    
    def __init__(self, config):
        #TODO Sinnvolle self parameter
        super(gymFMU, self).__init__()
        self.config = config

        for name in section_names:
            self.__dict__.update(config.get(name))

        # simulation times
        if round((self.stopTime - self.startTime) % self.dt, 9) != 0:
            self.stopTime = int(self.stopTime/self.dt) * self.dt
            print(f'Incompatible sample time and stop time.\n Using {self.stopTime} as stop time instead')
        if round(self.actionInterval % self.dt, 9) != 0:
            self.actionInterval = int(self.actionInterval/self.dt) * self.dt
            print(f'Incompatible sample time and action interval.\n Using {self.actionInterval} as interval instead')

        # steps
        self.numSteps = int(self.actionInterval/self.dt)
        self.totalSteps = int((self.stopTime - self.startTime) / self.dt)

        # initialize FMU
        self.fmu = FMU(self.config)
        self.observation_space = self.getObservationSpace()
        self.action_space = self.getActionSpace()

    def reset(self):        
        self.time = self.startTime 
        self.stepCount = 0
        self.resetIO()

        # reset FMU        
        self.fmu.resetFMU()

        # initial observation
        self._nextObservation()
        observation = self.outputs[self.stepCount,:]
        observation = self.obsProcessing(observation)

        if self.isDiscrete:
            return observation.astype(np.int64)[0]
        else:
            return observation

    def resetIO(self):
        self.inputs = np.empty([self.totalSteps, self.fmu.getNumInput()])
        self.outputs = np.empty([self.totalSteps+1, self.fmu.getNumOutput()])
        self.times = np.empty([self.totalSteps+1, 1])
        self.times[0] = self.startTime
        self.stepCount = 0

    def _nextObservation(self):
        # simulate fmu for iteration
        if DEBUG: print('###########################################')
        if DEBUG: print(f'Starting simulation at simulation time {self.time} [s]')

        for i in np.arange(self.numSteps):
            #with contextlib.redirect_stdout(None):
            # the inputs could change here
            self.additionalStepInput()

            # do one time step in fmu
            self.FMUstep()
            self.time += self.dt
            self.stepCount += 1
            self.times[self.stepCount] = self.time
            
            # save outputs
            self.outputs[self.stepCount,:] = np.array([self.fmu.fmu.getReal([x.valueReference])[0] for x in self.fmu.getOutput()])
                    
        if DEBUG: print('Simulation for current step done.')

    def FMUstep(self):
        #TODO self.time oder +stepTime
        self.fmu.fmu.doStep(currentCommunicationPoint=(self.time), communicationStepSize=self.dt)
        
    def step(self, action):
        #TODO store on drive?
        self._assignAction(action)

        self._nextObservation()
        
        # get observation vector
        observation = self.outputs[self.stepCount,:]
        
        # calculate rewards and set done flag
        done, reward = self.getReward(action, observation)
        observation = self.obsProcessing(observation)

        # end of simulation time reached?
        if self.time +self.dt > self.stopTime:
            done = True
            if LOG: print('Simulation done')
        
        return observation, reward, done, {}
        
    def close(self):
        self.fmu.closeFMU()

    def obsProcessing(self, observation):        
        return observation 
        
    #-------------------------------------------------------------------------------------
    # getter/setter-functions
    #-------------------------------------------------------------------------------------
        
    # write results to pickle file
    def exportResults(self):        
        pass

# ----------------------------------------------------------------------------
import _functions
DEBUG = False
LOG = False

for name in ['getActionSpace', 'getObservationSpace', 'resetIO', 
        'getReward', 'getMetric', '_assignAction', 'additionalStepInput', 'exportResults', 'obsProcessing']:
    try:
        setattr(gymFMU, name, getattr(_functions, name))
    except:
        print(f'No function for {name} implemented')





