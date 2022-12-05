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
        self.totalSteps = int((self.stopTime - self.startTime) / self.dt) + 1

        if self.totalSteps < self.numSteps:
            self.numSteps = self.totalSteps - 1

        # initialize FMU
        self.fmu = FMU(self.config)
        self.observation_space = self.getObservationSpace()
        self.action_space = self.getActionSpace()

        # dict for custom vars
        self.customVars = {}

    def reset(self):        
        self.time = self.startTime 
        self.stepCount = 0
        self.resetIO()

        # reset FMU        
        self.FMUstates = {}
        self.fmu.resetFMU()
        return self._reset() 

    def resetIO(self):
        self.inputs = np.empty([self.totalSteps, self.fmu.getNumInput()])
        self.outputs = np.empty([self.totalSteps, self.fmu.getNumOutput()])
        self.times = np.empty([self.totalSteps, 1])
        self.times[0] = self.startTime
        self.stepCount = -1

    def _nextObservation(self, steps=-1):
        # simulate fmu for iteration
        if DEBUG: print('###########################################')
        if DEBUG: print(f'Starting simulation at simulation time {self.time} [s]')
        if steps == -1:
            steps = self.numSteps

        for i in np.arange(steps):
            # the inputs could change here
            self.FMUstep()

            # do one time step in fmu
            self._FMUstep()

            # save outputs
            self.times[self.stepCount] = self.time
            self.inputs[self.stepCount,:] = np.array([self.fmu.fmu.getReal([x.valueReference])[0] for x in self.fmu.getInput()])
            self.outputs[self.stepCount,:] = np.array([self.fmu.fmu.getReal([x.valueReference])[0] for x in self.fmu.getOutput()])
                    
        if DEBUG: print('Simulation for current step done.')

    def _FMUstep(self):
        #TODO self.time oder +stepTime
        self.fmu.fmu.doStep(currentCommunicationPoint=(self.time), communicationStepSize=self.dt)
        self.stepCount += 1
        self.time += self.dt
        
    def step(self, action):
        #TODO store on drive?
        self._assignAction(action)

        self._nextObservation()
        
        # get observation vector
        observation = self.obsProcessing(self.outputs[self.stepCount,:])
        
        # calculate rewards and set done flag
        observation, reward, done = self.getReward(action, observation)

        #if reward < -1e10:
            #_ = self.reset()
            #observation, reward, done, _ = self.step(action)

        # end of simulation time reached?
        if self.time > self.stopTime + 0.5 * self.dt:
            done = True
            if LOG: print('Simulation done')
        
        return observation, reward, done, {}

        
    def close(self):
        self.fmu.closeFMU()

    def obsProcessing(self, observation):        
        return observation 

    # write results to pickle file
    def exportResults(self):        
        pass

    def saveRollbackState(self):
        self.FMUstate[str(self.currentStep)] = [self.fmu.fmu.getFMUstate(),
                                                self.inputs,
                                                self.outputs,
                                                self.times,
                                                self.stepCount,
                                                ]

        
    def makeRollback(self, step):                
        print('performing rollback')
        self.fmu.fmu.setFMUstate(state = self.FMUstates[str(step)][0])
        self.inputs, self.outputs, self.times, self.stepCoutn = self.FMUstates[str(step)][1:]
        
        
    #-------------------------------------------------------------------------------------
    # getter/setter-functions
    #-------------------------------------------------------------------------------------
        

# ----------------------------------------------------------------------------
import _functions
DEBUG = False
LOG = False

for name in ['getActionSpace', 'getObservationSpace', 'resetIO', 'FMUstep', '_assignAction',
        'getReward', 'getMetric', '_reset', 'exportResults', 'obsProcessing']:
    try:
        setattr(gymFMU, name, getattr(_functions, name))
    except:
        print(f'No function for {name} implemented')





