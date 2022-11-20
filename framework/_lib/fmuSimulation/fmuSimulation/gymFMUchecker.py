import fmuSimulation.gymFMU
import numpy as np

class FMUchecker(fmuSimulation.gymFMU.gymFMU):
    def __init__(self, config):
        super().__init__(config)
        self.time = self.startTime
        self.stepCount = 0
        self.outputs = np.empty([self.totalSteps+1, self.fmu.getNumOutput()])
        self.times = np.empty([self.totalSteps+1, 1])

    def setAction(self, action):
        if not self.checkAction(action):
            print('Error. Action is not in actionspace')
        self.assignAction(action)


    def getObservation(self):
        self._nextObservation()
        observation = self.outputs[self.stepCount,:]

        if not self.checkObservation(observation.astype(np.float32)):
            print('Error. Observation is not in observationspace')
            print(f'Observation: {observation} with type {type(observation)} \n \
                  Observation space: {self.getObservationSpace()}')
        return observation


    def checkObservation(self, observation):
        return self.getObservationSpace().contains(observation)      

    def checkAction(self, action):
        return self.getActionSpace().contains(action)


    #def checkActionSpace(self):

