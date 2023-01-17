# Author Robert Annuth - robert.annuth@tuhh.de

from fmpy.fmi2 import FMU2Slave
from fmpy import read_model_description, extract
import shutil
import logging
logger = logging.getLogger(__name__)


# config sections accessed by this class
section_names = 'FMU', 'General'


class FMU:
    """
    this class handels the interaction with the FMU object to seperate it from the gymnasium environment
    """

    def __init__(self, config):
        for name in section_names:
            self.__dict__.update(config.get(name))

        self.readFMU()
        logger.debug('Initializing FMU')
        self.fmu.instantiate()
        self.initFMU()
        logger.debug('Reading IOs FMU')
        self.getIO()

    # unzip and open FMU
    def readFMU(self):
        logger.debug('Starting to read FMU')
        logger.info('Using: {}'.format(self.fmuPath))
        self.description = read_model_description(self.fmuPath)
        self.zipDir = extract(self.fmuPath)
        self.fmu = FMU2Slave(guid=self.description.guid,
                             unzipDirectory=self.zipDir,
                             modelIdentifier=self.description.coSimulation.modelIdentifier,
                             instanceName='instance0')
        logger.debug('Success creating FMU object')
        logger.info('Unzipped in {}'.format(self.zipDir))

    # initialize FMU
    def initFMU(self):
        self.fmu.setupExperiment(startTime=self.startTime, tolerance=1e-6,
                                 stopTime=self.stopTime+self.dt)
        # set initial inputs
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

    # reset FMU to initial state
    def resetFMU(self):
        logger.debug('Initializing FMU')
        self.fmu.reset()
        self.initFMU()

    # get IO description of FMU
    def getIO(self):
        self.input = [x for x in self.description.modelVariables
                      if x.name.startswith(tuple(['I_', 'Control']))]
        self.output = [x for x in self.description.modelVariables
                       if x.name.startswith(tuple(['O_', 'Measurement']))]
        self.vars = [x for x in self.description.modelVariables
                     if not x.name.startswith(tuple(['O_', 'I_', 'Control', 'Measurement']))]

        # put in same order as GUI
        self.input = self.input[::-1]
        self.output = self.output[::-1]
        self.vars = self.vars[::-1]

        self.Inames = [x.name for x in self.input]
        self.Onames = [x.name for x in self.output]

        logger.debug('\nFound Inputs:')
        for i, x in enumerate(self.input):
            logger.debug('{}: {}'.format(i, x))
        logger.debug('\nFound Outputs:')
        for i, x in enumerate(self.output):
            logger.debug('{}: {}'.format(i, x))
        logger.debug('\nFound Vars:')
        for i, x in enumerate(self.vars):
            logger.debug('{}: {}'.format(i, x))

    # terminate FMU after simulation
    def closeFMU(self):
        logger.info('Close fmu and try deleting unzipdir')
        self.fmu.terminate()
        self.fmu.freeInstance()
        shutil.rmtree(self.zipDir, ignore_errors=True)

    # -------------------------------------------------------------------------------------
    # getter/setter-functions
    # -------------------------------------------------------------------------------------

    def getInput(self):
        return self.input

    def getNumInput(self):
        return int(len(self.input))

    def getOutput(self):
        return self.output

    def getNumOutput(self):
        return int(len(self.output))
