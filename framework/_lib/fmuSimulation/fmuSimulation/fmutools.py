
#-------------------------------------------------------------------------------------
# FMU functions
#-------------------------------------------------------------------------------------
# DEBUG = os.getenv("DEBUG", False) is not None
DEBUG = False
# LOG = os.getenv("LOG", False) is not None
LOG = False

import shutil
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave

section_names = 'FMU', 'General'
    
class FMU:
    def __init__(self, config):
        for name in section_names:
            self.__dict__.update(config.get(name))
        
        self.readFMU()
        self.initFMU()      

    # read FMU
    def readFMU(self):
        if DEBUG: print('Starting to read FMU')
        if LOG: print('Using: {}'.format(self.fmuPath))
        self.description = read_model_description(self.fmuPath)
        self.zipDir = extract(self.fmuPath)
        self.fmu = FMU2Slave(guid=self.description.guid,
                             unzipDirectory=self.zipDir,
                             modelIdentifier=self.description.coSimulation.modelIdentifier,
                             instanceName='instance0')
        if DEBUG: print('Success creating FMU object')
        if LOG: print('Unzipped in {}'.format(self.zipDir))

    # initialize FMU
    def initFMU(self):
        if DEBUG: print('Initializing FMU')
        self.fmu.instantiate()
        self.fmu.setupExperiment(startTime=self.startTime, tolerance=1e-6, 
                stopTime=self.stopTime)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()
        if DEBUG: print('Reading IOs FMU')
        self.getIO()

    # reset FMU to initial state
    def resetFMU(self):
        if DEBUG: print('Initializing FMU')
        self.fmu.reset()
        self.fmu.setupExperiment(startTime=self.startTime, tolerance=self.tolerance,
                stopTime=self.stopTime)
        
        # set initial inputs
        self.fmu.enterInitializationMode()

        self.fmu.exitInitializationMode()

    # get IO description of FMU
    def getIO(self):
        self.input = [x for x in self.description.modelVariables 
                if x.name.startswith(tuple(['I_','Control']))]
        self.output = [x for x in self.description.modelVariables 
                if x.name.startswith(tuple(['O_','Measurement']))]
        self.vrs = [x for x in self.description.modelVariables 
                if not x.name.startswith(tuple(['O_','I_','Control','Measurement']))]

        self.Inames = [x.name for x in self.input]
        self.Onames = [x.name for x in self.output]

        if LOG: 
            print('\nFound Inputs:')
            for i, x in enumerate(self.input):
                print('{}: {}'.format(i,x))
            print('\nFound Outputs:')
            for i, x in enumerate(self.output):
                print('{}: {}'.format(i,x))
            print('\nFound Vars:')
            for i, x in enumerate(self.vrs):
                print('{}: {}'.format(i,x))

    # terminate FMU after simulation
    def closeFMU(self):
        if LOG: print('Close fmu and try deleting unzipdir')
        self.fmu.terminate()
        self.fmu.freeInstance()
        shutil.rmtree(self.zipDir, ignore_errors=True)

    #-------------------------------------------------------------------------------------
    # getter/setter-functions
    #-------------------------------------------------------------------------------------
    def getInput(self):
        return self.input

    def getNumInput(self):
        return int(len(self.input))

    def getOutput(self):
        return self.output

    def getNumOutput(self):
        return int(len(self.output))
        
