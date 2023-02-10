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
    See https://https://gymnasium.farama.org/ for information about API and 
    necessary functions. The attributes and methods are sorted by decreasing 
    importance for user implementation. Instanitate this class for the RL agent.

    Short Guide
    -----------
    1. Create Simulink FMU using the README guide
    2. Create config with all relevant information
    3. Create child class and define

        Required:
        - Your custom reward function

        Optional:
        - Restrict the observation or action space
        - You could define special observation postprocessing 
        - You can specify additional environment inputs beside 
          the agents action
        - Export your results
        - Define rollback situations

    4. Let the agent do its job!


    Attributes
    ----------
    config : dict
        Dictionary containing all config variables.
    steps_between_actions : int
        Amount of FMU (environment) steps between agent actions (min 1).
    steps_simulation : int
        Total steps of the simulation.
    fmu : class fmutools
        FMU simulation object .
    observation_space : gymnasium.space
        Observation space.
    action_space : gymnasium.space
        Action space.
    FMUstate : list
        Containing all saved FMU states for rollbacks.
    seed : int
        The seed can be used for non deterministic implementations but is not 
        considered for the FMU, since the simulation is fully deterministic.


    Methods
    -------
    __init__(config):
        Initialize class with config dictionary.
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
    FMUstep_():
        During the FMU simulation the environment could have internal dynamics 
        modelled in python. Those can be set here.
    assignAction_(action):
        Assign action of the agent to the FMU.
    get_reward_():
        Calculate reward for an action.
    close():
        Close FMU and clean up.
    export_results():
        Enables users to export results.
    save_rollbackstate():
        Save environment state enabling rollbacks.
    perform_rollback(step):
        Perform rollback and return to previous environment state.
    """
    # FMUstep(self):

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
        The space is defined with respect to the loaded FMU but can be restricted.

        Returns
        Parameters
        ----------
        additional : str, optional
            More info to be displayed (default is None)

        Returns
        -------
        space : gymnasium.space
            Space defining action space of the agent
        """
        # TODO Implement
        pass

    def get_observation_space(self):
        """
        The space is defined with respect to the loaded FMU but can be restricted.

        Returns
        -------
        space : gymnasium.space
            Space defining observation space of the agent
        """
        # TODO Implement
        pass

# ----------------------------------------------------------------------------
# Reset
# ----------------------------------------------------------------------------
    def reset(self, seed=None):
        """
        Default reset function for gymnasium class

        Parameters
        ----------
        seed : int, optional
            The seed is not used for the FMU since those calculations are deterministic 
            but could be used by the user e.g. for weather models interacting with 
            the FMU during the simulation
        """
        self.seed = seed
        self.time = self.start_time
        self.step_count = 0
        self.resetIO()

        # reset FMU
        self.FMU_states = {}
        self.fmu.resetFMU()
        # calling internal reset function which can be overwritten
        # and allows customization
        return self.reset_(seed)

    def resetIO(self):
        """
        Resetting lists contianing the inputs / outputs and actions of each step 
        and the internal variables.
        """
        self.inputs = np.empty([self.steps_simulation, self.fmu.getNumInput()])
        self.outputs = np.empty(
            [self.steps_simulation, self.fmu.getNumOutput()])
        self.times = np.empty([self.steps_simulation, 1])
        self.times[0] = self.start_time
        self.step_count = -1

    def reset_(self, seed=None):
        """
        This internal reset function provides an interface to modify the 
        environment at every reset. You can overwrite this!
        The code could also depend on the seed and it is possible to modify the 
        returned observation

        Parameters
        ----------
        seed : int, optional
            see reset function
        """
        # TODO return missing
        pass
        return observation

# ----------------------------------------------------------------------------
# Step
# ----------------------------------------------------------------------------
    def step(self, action):
        # assign actions to FMU input
        self.assignAction(action)

        # simulate next step(s) of the FMU
        self._nextObservation()

        # get observation vector / outputs of the FMU
        observation = self.obsProcessing(self.outputs[self.step_count, :])

        # calculate rewards and set done flag
        reward, truncated, info = self.get_reward(action, observation)

        # end of simulation time reached?
        if self.time > self.stop_time + 0.5 * self.dt:
            terminated = True
            logger.info('Simulation done')

        return observation, reward, terminated, truncated, info

    def assignAction(self, action):
        # assign actions to inputs
        # TODO use for loop and check with action space 
        self.fmu.fmu.setReal([self.fmu.input[0].valueReference], [action[0]])

    def _nextObservation(self, steps=-1):
        # simulate fmu for the specified amount of steps 
        logging.debug(
            f'Starting simulation at simulation time {self.time} [s]')

        # only during reset the amount of steps is set
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
    
    def FMUstep_(self):
        """
        This function is called before each FMU step. Here you can set FMU inputs
        independent of the agent action. This could be used e.g. for weather data
        influencing the FMU simulation
        
        Use the code below to access the FMU inputs.
        self.fmu.fmu.setReal([self.fmu.input[0].valueReference], [action[0]])
        """
        pass

    def get_reward(self, observation):
        info = {}
        reward = 1
        terminated = False
        truncated = False
        return reward, truncated, info


# ----------------------------------------------------------------------------
# Close / Export / Rollback
# ----------------------------------------------------------------------------

    def close(self):
        """
        Close FMU and clean up temporary data
        """
        self.fmu.closeFMU()

    def export_results(self):
        """
        This can be used to export results 
        """
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