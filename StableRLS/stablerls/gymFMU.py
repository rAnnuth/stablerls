# Author Robert Annuth - robert.annuth@tuhh.de

import types
import contextlib
import json
import gymnasium as gym
import numpy as np
from stablerls.fmutools import FMU
from stablerls.configreader import smart_parse
import logging

logger = logging.getLogger(__name__)


# specify parameter sections available in the StableRLS class
# ----------------------------------------------------------------------------
section_names = "Reinforcement Learning", "General", "FMU"
# ----------------------------------------------------------------------------


class StableRLS(gym.Env):
    """Custom environment for simulation of FMUs with gymnasium interface. See
    https://gymnasium.farama.org/ for information about the API and
    necessary functions. Instantiate this class for the RL agent.  Short Guide:

    1. Create Simulink FMU using the README guide
    2. Create -file-.cfg (config) with all relevant information
    3. Create a child class and let the agent do its job!

    Required:

    - Your custom reward function

    Optional:

    - Define your own, restricted observation or action spaces
    - Define your own customized observation post-processing
    - Specify additional environment inputs beside the agent's action

    - Export your results
    - Define rollback situations

    Attributes
    ----------
    config : dict
        Dictionary containing all config variables.
    """

    # ----------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------
    def __init__(self, config):
        """Constructor method."""
        super(StableRLS, self).__init__()
        self.config = config

        # make parameters of config file available as class parameters
        for name in section_names:
            self.__dict__.update(config.get(name))
        # set missing parameters to default values
        # by default, the agent chooses one action per time step
        if not hasattr(self, "action_interval"):
            self.action_interval = self.dt
        # set start time of the simulation
        if not hasattr(self, "start_time"):
            self.start_time = 0.0
        # specify if the inputs are set to zero at each reset() call
        if not hasattr(self, "reset_inputs"):
            self.reset_inputs = True

        # check config settings for simulation time
        if not round((self.stop_time - self.start_time) / self.dt, 9).is_integer():
            self.stop_time = int(self.stop_time / self.dt) * self.dt
            logger.warning(
                f"Incompatible time step and stop time.\n Using {self.stop_time} as"
                " stop time instead"
            )
        if not round(self.action_interval / self.dt, 9).is_integer():
            self.action_interval = int(self.action_interval / self.dt) * self.dt
            logger.warning(
                "Incompatible time step and action interval.\n Using"
                f" {self.action_interval} as interval instead"
            )

        # calculate number of simulation steps
        self.steps_between_actions = int(self.action_interval / self.dt)
        self.steps_simulation = int((self.stop_time - self.start_time) / self.dt) + 1

        # change action interval to (simulation steps - 1)*dt if it leads to too many steps
        # between actions
        if self.steps_simulation < self.steps_between_actions:
            self.steps_between_actions = self.steps_simulation - 1
            self.action_interval = self.steps_between_actions * self.dt
            logger.warning(
                "Action interval inconsistent with simulation duration.\n Using"
                f" {self.action_interval} as interval instead"
            )

        # initialize FMU
        self.fmu = FMU(self.config)
        self.observation_space = self.set_observation_space()
        self.action_space = self.set_action_space()

        # dict for custom vars
        self.info = {}

    def set_action_space(self):
        """Setter function for the action space of the agent. As default,
        the action space is defined with respect to the loaded FMU and 
        uses all inputs as actions.
        For other (restricted) action spaces, this function needs to be 
        modified. For information about possible space definitions, see
        https://gymnasium.farama.org/api/spaces/.

        Returns
        -------
        space : gymnasium.space
            Returns the unbounded action space defined by the FMU inputs
        """
        high = np.arange(len(self.fmu.input)).astype(np.float32)
        high[:] = np.inf
        low = high * -1
        return gym.spaces.Box(low, high)

    def set_observation_space(self):
        """Setter function for the observation space of the agent. As 
        default, the action space is defined with respect to the loaded FMU 
        and uses all outputs as observations.
        For other (restricted) observation spaces, this function needs to be 
        modified. For information about possible space definitions, see
        https://gymnasium.farama.org/api/spaces/.

        Returns
        -------
        space : gymnasium.space
            Returns the unbounded observation space defined by the FMU 
            outputs
        """
        high = np.arange(len(self.fmu.output)).astype(np.float32)
        high[:] = np.inf
        low = high * -1
        return gym.spaces.Box(low, high)

    # ----------------------------------------------------------------------------
    # Reset
    # ----------------------------------------------------------------------------
    def reset(self, seed=None):
        """Default reset function for gymnasium class.

        Parameters
        ----------
        seed : int, optional
            The seed is not used for the FMU since those calculations are 
            deterministic but could be used by the user e.g. for weather models 
            interacting with the FMU during the simulation.

        Returns
        -------
        observation : gymnasium.space
            Observation created during reset call (defined behavior by gym)
        """
        self.seed = seed
        self.time = self.start_time
        self.step_count = 0
        self._resetIO()

        # reset FMU
        self.FMU_states = {}
        self.fmu.resetFMU()
        # calling internal reset function which can be overwritten
        # and allows customization
        return self.reset_(seed)

    def _resetIO(self):
        """Resetting lists containing the inputs / outputs and actions of each
        step and the internal variables."""
        self.inputs = np.empty([self.steps_simulation, self.fmu.getNumInput()])
        self.outputs = np.empty([self.steps_simulation, self.fmu.getNumOutput()])
        self.times = np.empty([self.steps_simulation, 1])
        self.times[0] = self.start_time
        self.step_count = -1

    def reset_(self, seed=None):
        """This internal reset function provides an interface to modify the
        environment at every reset. You can overwrite this!  The code could
        also depend on the seed and it is possible to modify the returned
        observation.

        The default behavior is to simulate the initial step and return all observed 
        values. However, the inputs are not reset therefore.

        Parameters
        ----------
        seed : int, optional
            The seed is not used for the FMU since its calculations are 
            deterministic but could be used by the user e.g. for weather models 
            interacting with the FMU during the simulation.
        """
        if self.reset_inputs:
            # set all inputs to zero for consistent simulation results
            for x in self.fmu.input:
                self.fmu.fmu.setReal([x.valueReference], [0])

        # get the first observation as specified by gymnaisum
        self._next_observation(steps=1)
        return self.obs_processing(self.outputs[self.step_count, :])

    # ----------------------------------------------------------------------------
    # Step
    # ----------------------------------------------------------------------------
    def step(self, action):
        """Run one time step of the environment's dynamics using the agent
        actions.  (Adapted from gymnasium documentation v0.28.1)

        Parameters
        ----------
        action : list
            An action provided by the agent to update the environment state.

        Returns
        ----------
        observation : ObsType
            An element of the environment's `observation_space` as the next 
            observation due to the agent actions.  An example is a numpy array 
            containing the positions and velocities of the pole in CartPole.
        reward : SupportsFloat
            The reward as a result of taking the action.
        terminated : bool
            Whether the agent reaches the terminal state (as defined under the 
            MDP of the task) which can be positive or negative. An example is 
            reaching the goal state or moving into the lava from the Sutton and 
            Barton, Gridworld. If true, the user needs to call :meth:`reset`.
        truncated : bool
            Whether the truncation condition outside the scope of the MDP is 
            satisfied. Typically, this is a timelimit, but could also be used 
            to indicate an agent physically going out of bounds. Can be used to 
            end the episode prematurely before a terminal state is reached. If 
            true, the user needs to call :meth:`reset`.  
        info : dict 
            Contains auxiliary diagnostic information (helpful for debugging, learning, 
            and logging).  This might, for instance, contain: metrics that describe 
            the agent's performance state, variables that are hidden from observations, 
            or individual reward terms that are combined to produce the total reward.  
            In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish 
            truncation and termination, however this is deprecated in favor of 
            returning terminated and truncated variables.
        """
        # make sure we have a float32 numpy array
        action = np.array(action).astype(np.float32)
        # assign actions to FMU input
        self.assignAction(action)

        # simulate next step(s) of the FMU
        self._next_observation()

        # get observation vector / outputs of the FMU
        observation = self.obs_processing(self.outputs[self.step_count, :])

        # calculate rewards and if needed set truncated flag
        reward, terminated, truncated, info = self.get_reward(action, observation)

        # end of simulation time reached?
        if self.time > self.stop_time + 0.5 * self.dt:
            truncated = True
            logger.info("Simulation done")

        return observation, reward, terminated, truncated, info

    def assignAction(self, action):
        """Assign actions to the inputs of the FMU/environment.

        Parameters
        ----------
        action : list
            An action provided by the agent to update the environment state.
        """
        # assign actions to inputs
        # check if actions are within action space
        if not self.action_space.contains(action):
            logger.info(f"The actions are not within the action space. Action: {action}. Time: {self.time}")

        # assign actions to the action space
        for i, val in enumerate(action):
            if val == None:
                continue
            self.fmu.fmu.setReal([self.fmu.input[i].valueReference], [val])

    def _next_observation(self, steps=-1):
        """It might be required to run the simulation for multiple steps
        between the inteactions of the agent.  By default the agent only
        observes state after the last of the simulation.

        Parameters
        ----------
        steps : int, optional
            Number of steps the FMU model should run
        """

        # simulate fmu for the specified amount of steps
        logging.debug(f"Starting simulation at simulation time {self.time} [s]")

        # only during reset the amount of steps is set
        if steps == -1:
            steps = self.steps_between_actions

        for _ in np.arange(steps):
            # inputs of the FMU can changed independend of the agent
            self.FMU_external_input()

            # simulate FMU for one time step (dt)
            self._FMUstep()

            # save simulation step results
            self.times[self.step_count] = self.time
            self.inputs[self.step_count, :] = np.array(
                [self.fmu.fmu.getReal([x.valueReference])[0] for x in self.fmu.input]
            )
            self.outputs[self.step_count, :] = np.array(
                [self.fmu.fmu.getReal([x.valueReference])[0] for x in self.fmu.output]
            )

        logging.debug("Simulation for current step done.")

    def _FMUstep(self):
        """This internal step function handles the interaction with the FMU."""
        self.fmu.fmu.doStep(
            currentCommunicationPoint=(self.time), communicationStepSize=self.dt
        )
        self.step_count += 1
        self.time += self.dt

    def FMU_external_input(self):
        """This function is called before each FMU step. Here you can set FMU
        inputs independent of the agent action. This could be used e.g. for
        weather data influencing the FMU simulation.

        Use the code below to access the FMU inputs.
        self.fmu.fmu.setReal([self.fmu.input[0].valueReference], [value])
        """
        pass

    def obs_processing(self, observation):
        """If the agent is supposed to observe modified values the simulated
        values can be modified here before the reward calculation.

        Parameters
        ----------
        observation : ObsType
            The observation by the FMU/ environment

        Returns
        -------
        observation : ObsType
            Per default only returns the input but by overwriting the function custom requirements can be met
        """
        return observation

    def get_reward(self, action, observation):
        """The reward function depends on the specific usecase and must be
        specified by the user.

        Parameters
        ----------
        action: ActionType
            The initial action leads to the observed state of the FMU/ environment
        observation : ObsType
            The modified observation by the FMU/ environment (:meth:`obs_processing`)

        Returns
        -------
        reward : float
            Calculated reward for the given action and observation
        terminated : bool
            Set flag if episode should be terminated. It is automatically 
            terminated if the maximum time is reached
        truncated : bool
            Set flag if the agent is truncated
        info : dict
            Info dict which is empty by default
        """
        info = {}
        reward = 1
        terminated = False
        truncated = False
        return reward, terminated, truncated, info

    # ----------------------------------------------------------------------------
    # Close / Export / Rollback
    # ----------------------------------------------------------------------------

    def close(self):
        """Close FMU and clean up temporary data.

        This should be called after the simulation ends.
        """
        self.fmu.closeFMU()

    def export_results(self):
        """This function can be overwritten to acess and export results of the
        agent."""
        pass

    def _save_rollbackstate(self):
        """Currently matlabs limited export capabilities prevent rollbacks if
        they enable this at any time this will work.

        Therefore, the function is currently marked as private.
        """
        logging.info(f"Creating rollback state at time step {self.step_count}")
        # get the current state
        state = self.fmu.fmu.getFMUstate()
        # serialize the state
        serialized_state = self.fmu.fmu.serializeFMUstate(state)
        self.FMU_states[str(self.step_count)] = [
            state,
            serialized_state,
            self.inputs.copy(),
            self.outputs.copy(),
            self.times.copy(),
            self.step_count,
        ]

    def _perform_rollback(self, step):
        """Currently matlabs limited export capabilities prevent rollbacks if
        they enable this at any time this will work.

        Therefore, the function is currently marked as private.
        """
        logger.info(f"Performing rollback to the state at step {step}")
        logger.info(
            "If MATLAB created this FMU the rollback did not affect the FMU state"
        )
        # de-serialize the state
        deserialized_state = self.fmu.fmu.deSerializeFMUstate(
            self.FMU_states[str(step)][1], self.FMU_states[str(step)][0]
        )
        # set the state
        self.fmu.fmu.setFMUstate(deserialized_state)
        # free memory
        self.fmu.fmu.freeFMUstate(deserialized_state)
        # self.fmu.fmu.setFMUstate(state=self.FMU_states[str(step)][0])
        self.inputs, self.outputs, self.times, self.step_count = self.FMU_states[
            str(step)
        ][2:]
