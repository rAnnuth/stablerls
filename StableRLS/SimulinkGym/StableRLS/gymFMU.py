# Author Robert Annuth - robert.annuth@tuhh.de
import types
import contextlib
import json
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from SimulinkGym.fmutools import FMU
from SimulinkGym.configReader import smartParse
import logging

logger = logging.getLogger(__name__)


# specify parameter sections available in the stableRLS class
# ----------------------------------------------------------------------------
section_names = 'Reinforcement Learning', 'General', 'FMU'
# ----------------------------------------------------------------------------


class stableRLS(gym.Env):
    '''
    Custom environment for simulation of FMUs with gymnasium interface
    See https://https://gymnasium.farama.org/ for information about API and necessary functions.

    Class Parameters:
    '''

# ----------------------------------------------------------------------------
# Initialization
# ----------------------------------------------------------------------------
    def __init__(self, config):
        super(stableRLS, self).__init__()
        self.config = config

        # make parameters of config file available as class parameters
        for name in section_names:
            self.__dict__.update(config.get(name))

        # check config settings for simulation time
        if round((self.stop_time - self.start_time) % self.dt, 9) != 0:
            self.stop_time = int(self.stop_time/self.dt) * self.dt
            logger.warning(
                f'Incompatible sample time and stop time.\n Using {self.stop_time} as stop time instead')
        if round(self.actionInterval % self.dt, 9) != 0:
            self.actionInterval = int(self.actionInterval/self.dt) * self.dt
            logger.warning(
                f'Incompatible sample time and action interval.\n Using {self.actionInterval} as interval instead')

        # calculate steps of simulation
        self.steps_between_actions = int(self.actionInterval/self.dt)
        self.simulation_steps = int(
            (self.stop_time - self.start_time) / self.dt) + 1

        if self.simulation_steps < self.steps_between_actions:
            self.steps_between_actions = self.simulation_steps - 1

        # initialize FMU
        self.fmu = FMU(self.config)
        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()

        # dict for custom vars
        self.info = {}

    def get_action_space(self):
        # TODO Implement
        pass

    def get_observation_space(self):
        # TODO Implement
        pass

# ----------------------------------------------------------------------------
# Reset
# ----------------------------------------------------------------------------
    def reset(self, seed=None):
        self.time = self.start_time
        self.step_count = 0
        self.resetIO()

        # reset FMU
        self.FMU_states = {}
        self.fmu.resetFMU()
        # calling internal reset function which can be overwritten and allows customization
        return self._reset(seed)

    def _reset(self, seed=None):
        """
        This internal reset function provides an interface to modify the environment at every reset. 
        The code could also depend on the seed and it is possible to modify the returned observation
        """
        # TODO return missing
        pass
        return observation

    def resetIO(self):
        """
        Resetting lists contianing the inputs / outputs and actions of each step and the internal variables.
        """
        self.inputs = np.empty([self.simulation_steps, self.fmu.getNumInput()])
        self.outputs = np.empty(
            [self.simulation_steps, self.fmu.getNumOutput()])
        self.times = np.empty([self.simulation_steps, 1])
        self.times[0] = self.start_time
        self.step_count = -1

# ----------------------------------------------------------------------------
# Step
# ----------------------------------------------------------------------------
    def _nextObservation(self, steps=-1):
        # simulate fmu for iteration
        logging.debug('###########################################')
        logging.debug(
            f'Starting simulation at simulation time {self.time} [s]')
        if steps == -1:
            steps = self.steps_between_actions

        for i in np.arange(steps):
            # the inputs could change here
            self.FMUstep()

            # do one time step in fmu
            self._FMUstep()

            # save outputs
            self.times[self.step_count] = self.time
            self.inputs[self.step_count, :] = np.array(
                [self.fmu.fmu.getReal([x.valueReference])[0] for x in self.fmu.getInput()])
            self.outputs[self.step_count, :] = np.array(
                [self.fmu.fmu.getReal([x.valueReference])[0] for x in self.fmu.getOutput()])

        logging.debug('Simulation for current step done.')

    def _FMUstep(self):
        self.fmu.fmu.doStep(currentCommunicationPoint=(
            self.time), communicationStepSize=self.dt)
        self.step_count += 1
        self.time += self.dt

    def step(self, action):
        self._assignAction(action)

        self._nextObservation()

        # get observation vector
        observation = self.obsProcessing(self.outputs[self.step_count, :])

        # calculate rewards and set done flag
        observation, reward, done = self.getReward(action, observation)

        # if reward < -1e10:
        #_ = self.reset()
        #observation, reward, done, _ = self.step(action)

        # end of simulation time reached?
        if self.time > self.stop_time + 0.5 * self.dt:
            done = True
            logger.info('Simulation done')

        return observation, reward, done, {}

# ----------------------------------------------------------------------------
# Close / Rollback
# ----------------------------------------------------------------------------
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
                                                self.step_count,
                                                ]

    def makeRollback(self, step):
        logger.info('performing rollback')
        self.fmu.fmu.setFMUstate(state=self.FMU_states[str(step)][0])
        self.inputs, self.outputs, self.times, self.stepCoutn = self.FMU_states[str(
            step)][1:]
