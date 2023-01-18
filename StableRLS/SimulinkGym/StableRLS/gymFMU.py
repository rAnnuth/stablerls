# Author Robert Annuth - robert.annuth@tuhh.de
import types
import contextlib
import json
import gymnasium as gym
import numpy as np
from StableRLS.fmutools import FMU
from StableRLS.configreader import smart_parse
import logging

logger = logging.getLogger(__name__)


# specify parameter sections available in the stableRLS class
# ----------------------------------------------------------------------------
section_names = 'Reinforcement Learning', 'General', 'FMU'
# ----------------------------------------------------------------------------


class stableRLS(gym.Env):
    """
    Custom environment for simulation of FMUs with gymnasium interface
    See https://https://gymnasium.farama.org/ for information about API and necessary functions.

    Attributes
    ----------
    config : dict
        first name of the person
    steps_between_actions : int
        family name of the person
    steps_simulation: int
        age of the person
    action_interval : int
        xxx
    fmu : class fmutools
        xxx
    observation_space : 
        xxx
    action_space :
    xxx



    Methods
    -------

    get_action_space():
        Returns action space of the environment.
    get_observation_space():
        Returns observation space of the environment.
    reset(seed=None):
        Reset function for internal variables and FMU.

    reset_(seed=None):
        Reset function which can be modified by the user.
    resetIO():
        Resetting inputs and outputs of the FMU.
    step(action):
        Step function for the RL algorithm.
    _nextObservation(steps=-1):
        Simulate FMU until next action is required.
    _FMUstep():
        Calculate next FMU state.
    assignAction_(action):
        Assign action of the agent to the FMU.
    get_reward_():
        Calculate reward for an action.
    close():
        Close FMU and clean up.

    export_results(self):
    save_rollbackstate(self):
    perform_rollback(self, step):
    """
    #observation_processing(self, observation):
    #FMUstep(self):

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
        if round(self.action_interval % self.dt, 9) != 0:
            self.action_interval = int(self.action_interval/self.dt) * self.dt
            logger.warning(
                f'Incompatible sample time and action interval.\n Using {self.action_interval} as interval instead')

        # calculate steps of simulation
        self.steps_between_actions = int(self.action_interval/self.dt)
        self.steps_simulation = int(
            (self.stop_time - self.start_time) / self.dt) + 1

        if self.steps_simulation < self.steps_between_actions:
            self.steps_between_actions = self.steps_simulation - 1

        # initialize FMU
        self.fmu = FMU(self.config)
        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()

        # dict for custom vars
        self.info = {}

    def get_action_space(self):
        """
        Parameters
        ----------
        additional : str, optional
            More info to be displayed (default is None)

        Returns
        -------
        None
        """
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
        return self.reset_(seed)

    def reset_(self, seed=None):
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
        self.inputs = np.empty([self.steps_simulation, self.fmu.getNumInput()])
        self.outputs = np.empty(
            [self.steps_simulation, self.fmu.getNumOutput()])
        self.times = np.empty([self.steps_simulation, 1])
        self.times[0] = self.start_time
        self.step_count = -1

# ----------------------------------------------------------------------------
# Step
# ----------------------------------------------------------------------------
    def _nextObservation(self, steps=-1):
        # simulate fmu for iteration
        logging.debug(
            f'Starting simulation at simulation time {self.time} [s]')

        if steps == -1:
            steps = self.steps_between_actions

        for _ in np.arange(steps):
            # inputs of the FMU can changed independend of the agent
            self.FMUstep_()

            # simulate FMU for one timestep (dt)
            self._FMUstep()

            # save simulation step results
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

    def assignAction(self, action):
        # assign actions to inputs
        self.fmu.fmu.setReal([self.fmu.input[0].valueReference], [action[0]])

    def get_reward(self, observation):
        info = {}
        reward = 1
        terminated = False
        truncated = False
        return reward, truncated, info

    def step(self, action):
        self.assignAction(action)

        self._nextObservation()

        # get observation vector
        observation = self.obsProcessing(self.outputs[self.step_count, :])

        # calculate rewards and set done flag
        reward, truncated, info = self.get_reward(action, observation)

        # end of simulation time reached?
        if self.time > self.stop_time + 0.5 * self.dt:
            terminated = True
            logger.info('Simulation done')

        return observation, reward, terminated, truncated, info  

# ----------------------------------------------------------------------------
# Close / Rollback
# ----------------------------------------------------------------------------
    def close(self):
        self.fmu.closeFMU()

    def observation_processing(self, observation):
        # TODO do we need this
        return observation

    # write results to pickle file
    def export_results(self):
        pass

    def save_rollbackstate(self):
        self.FMUstate[str(self.currentStep)] = [self.fmu.fmu.getFMUstate(),
                                                self.inputs,
                                                self.outputs,
                                                self.times,
                                                self.step_count,
                                                ]

    def perform_rollback(self, step):
        logger.info(f'Performing rollback to the state at step {step}')
        self.fmu.fmu.setFMUstate(state=self.FMU_states[str(step)][0])
        self.inputs, self.outputs, self.times, self.stepCoutn = self.FMU_states[str(
            step)][1:]


# ----------------------------------------------------------------------------
# Other
# ----------------------------------------------------------------------------
def FMUstep(self):
    #use self.time to access simulation time and change input here
    #store in self.Inputs here
    #np.nditer() solves problems!
    pass

def exportResults(self):
    pass
