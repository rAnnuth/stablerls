# Author Robert Annuth - robert.annuth@tuhh.de

from fmpy.fmi2 import FMU2Slave
from fmpy import read_model_description, extract
import shutil
import logging

logger = logging.getLogger(__name__)

# config sections accessed by this class
section_names = "FMU", "General"


class FMU:
    """
    This class handels the interaction with the FMU object to separate it from the gymnasium environment.
    I should not be necessary to modify or interact with this.

    Attributes
    ----------
    config : dict
        Dictionary containing all config variables.

    """

    def __init__(self, config):
        """Constructor method"""
        # make parameters of config file available as class parameters
        for name in section_names:
            self.__dict__.update(config.get(name))
        # add missing parameters
        if not hasattr(self, "tolerance"):
            self.tolerance = 1e-6
        if not hasattr(self, "start_time"):
            self.start_time = 0

        # this is the default fmpy procedure
        self.readFMU()
        logger.debug("Initializing FMU")
        self.fmu.instantiate()
        self.initFMU()
        logger.debug("Reading IOs FMU")
        self.getIO()

    #
    def readFMU(self):
        """Unzip and open FMU file"""
        logger.debug("Starting to read FMU")
        logger.info("Using: {}".format(self.FMU_path))
        self.description = read_model_description(self.FMU_path)
        self.zipDir = extract(self.FMU_path)
        self.fmu = FMU2Slave(
            guid=self.description.guid,
            unzipDirectory=self.zipDir,
            modelIdentifier=self.description.coSimulation.modelIdentifier,
            instanceName="instance0",
        )
        logger.debug("Success creating FMU object")
        logger.info("Unzipped in {}".format(self.zipDir))

    # initialize FMU
    def initFMU(self):
        """Fmpy requires to setup an experiment before the simulation can be started"""
        self.fmu.setupExperiment(
            startTime=self.start_time, tolerance=1e-6, stopTime=self.stop_time + self.dt
        )
        # set initial inputs
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

    # reset FMU to initial state
    def resetFMU(self):
        """Reset the FMU to the inital state"""
        logger.debug("Initializing FMU")
        self.fmu.reset()
        self.initFMU()

    # get IO description of FMU
    def getIO(self):
        """Read all available inputs and outputs of the FMU"""
        self.output = [x.variable for x in self.description.outputs]
        self.input = [
            x for x in self.description.modelVariables if x not in self.output
        ]
        # remove time variable since we dont want to set it
        for i, x in enumerate(self.input):
            if x.name == "time":
                self.input.pop(i)
                break

        # put in correct order as GUI
        self.input = self.input[::-1]
        self.output = self.output[::-1]

        self.input_names = [x.name for x in self.input]
        self.output_names = [x.name for x in self.output]

        logger.info("Found inputs - access them by the corresponding number:")
        for i, x in enumerate(self.input_names):
            logger.info(" {}: {}".format(i, x))
        logger.info("Found outputs - access them by the corresponding number:")
        for i, x in enumerate(self.output_names):
            logger.info(" {}: {}".format(i, x))

    def closeFMU(self):
        """Terminate FMU after the simulation"""
        logger.info("Close fmu and try deleting unzipdir")
        self.fmu.terminate()
        self.fmu.freeInstance()
        shutil.rmtree(self.zipDir, ignore_errors=True)

    # -------------------------------------------------------------------------------------
    # getter/setter-functions
    # -------------------------------------------------------------------------------------

    def getInput(self):
        return self.input_names

    def getNumInput(self):
        return int(len(self.input))

    def getOutput(self):
        return self.output_names

    def getNumOutput(self):
        return int(len(self.output))
