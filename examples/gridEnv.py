import stablerls.gymFMU as gymFMU
import stablerls.configreader as cfg_reader
import stablerls.createFMU as createFMU
import gymnasium as gym
import numpy as np
import logging
import random

logger = logging.getLogger(__name__)

class GridEnv(gymFMU.StableRLS):
    def set_action_space(self):
        """Setter function for the action space of the agent. 
        This function overrides the base implementation of StableRLS to 
        choose only certain inputs as actions.

        Returns
        -------
        space : gymnasium.space
            Returns the action space defined by specified FMU inputs
        """
        return gym.spaces.MultiDiscrete([11, 11])
    
    def set_observation_space(self):
        """Setter function for the observation space of the agent. 
        This function overrides the base implementation of StableRLS to 
        choose only certain outputs as observations.

        Returns
        -------
        space : gymnasium.space
            Returns the observation space defined by specified FMU outputs
        """
        high = np.arange(8).astype(np.float32)
        high[:] = 1
        low = high * -1
        return gym.spaces.Box(low, high)
    
    def assignAction(self, action):
        """Changed assignment of actions to the FMU because only certain inputs
        are used for the agent actions.

        Parameters
        ----------
        action : list
            An action provided by the agent to update the environment state.
        """
        # assign actions to inputs
        # check if actions are within action space
        if not self.action_space.contains(action):
            logger.info(f"The actions are not within the action space. Action: {action}. Time: {self.time}")

        # convert discrete actions into steps of voltage references
        vStepGrid = (action[0] - 5) * 0.04    
        vStepBat = (action[1] - 5) * 0.04

        # add them to actual references to get new setpoints
        vRefGrid = self.fmu.fmu.getReal([self.fmu.input[3].valueReference])[0] + vStepGrid  
        vRefBat = self.fmu.fmu.getReal([self.fmu.input[5].valueReference])[0] + vStepBat

        # assign actions to the FMU inputs - take care of right indices!
        self.fmu.fmu.setReal([self.fmu.input[3].valueReference], [vRefGrid])
        self.fmu.fmu.setReal([self.fmu.input[5].valueReference], [vRefBat])

    def obs_processing(self, raw_obs):
        """Customised action processing: Only specific outputs are evaluated. Additionally,
        they are normalised to +-1.

        Parameters
        ----------
        raw_obs : ObsType
            The raw observation defined by all FMU outputs.
        Returns
        -------
        observation : ObsType
            The processed observation for the agent.
        """

        nDec = 2
        observation = np.array([  round((raw_obs[0] - self.nominal_voltage) / (0.1*self.nominal_voltage), 2),   # PV.V
                                  round((raw_obs[3] - self.nominal_voltage) / (0.1*self.nominal_voltage), 2),   # Grid.V                     
                                  round((raw_obs[6] - self.nominal_voltage) / (0.1*self.nominal_voltage), 2),   # Load.V
                                  round((raw_obs[13] - self.nominal_voltage) / (0.1*self.nominal_voltage), 2),  # Bat.V  
                                  round(raw_obs[2] / self.nominal_current, 2),                                  # PV.I
                                  round(raw_obs[5] / self.nominal_current, 2),                                  # Grid.I
                                  round(raw_obs[12] / self.nominal_current, 2),                                 # Bat.Inet
                                  round(raw_obs[11], nDec),                                                     # Bat.SOC
                               ]).astype(np.float32)

        return observation
    
    def reset_(self, seed=None):
        """Since zeros make no sense for the voltage references, the input reset is
        changed. Also, the initial SOC is chosen randomly between limits.

        Parameters
        ----------
        seed : int, optional
            - None -
        """
        # set voltage references to nominal voltage
        self.fmu.fmu.setReal([self.fmu.input[3].valueReference], [self.nominal_voltage])
        self.fmu.fmu.setReal([self.fmu.input[5].valueReference], [self.nominal_voltage])

        # set initial SOC, randomly chosen
        rand = random.random()
        if rand < 0.15:
            self.soc_init = 0.15
        elif rand > 0.85:
            self.soc_init = 0.85
        else:
            self.soc_init = rand 
        self.fmu.fmu.setReal([self.fmu.input[4].valueReference], [self.soc_init])

        # all other inputs are set during the calculation of the first step since 
        # they are external

        # get the first observation as specified by gymnaisum
        self._next_observation(steps=1)
        return self.obs_processing(self.outputs[self.step_count, :])

    def FMU_external_input(self):
        """This function is called before each FMU step. Here external FMU
        inputs independent of the agent action are set. In this case, this includes
        weather data and the load power.

        Use the code below to access the FMU inputs.
        self.fmu.fmu.setReal([self.fmu.input[0].valueReference], [value])
        """
        # Irradiance
        self.fmu.fmu.setReal([self.fmu.input[1].valueReference], [1000.0])

        # ModuleTemperature
        self.fmu.fmu.setReal([self.fmu.input[0].valueReference], [30.0])

        # LoadPower
        self.fmu.fmu.setReal([self.fmu.input[2].valueReference], [500.0])