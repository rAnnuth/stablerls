from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
import shutil, os, types, contextlib, pickle, datetime, time, sys
import default_control 

# DEBUG = os.getenv("DEBUG", False) is not None
DEBUG = False
# LOG = os.getenv("LOG", False) is not None
LOG = True

class sim:
  def __init__(self,simFile,stdout=None,dt=1e-3,stopTime=10,ctrlFkt=default_control.default_controller):
    """
    simFile - if *.slx a FMU is generated, if *.fmu simulation starts
    stdout - eg. 'out.txt' File used for shell output
    dt - simulation stepsize
    stopTime - simulation end time
    ctrlFkt - Function used to control plant
    The FMU generation requires specific folder structure
    """
    # set output if specified
    if not stdout == None:
      if LOG: print('Using {} as output file.'.format(stdout))
      sys.stdout = open(stdout, 'w')
    self.stopTime = stopTime

    # set up class variables
    self.name, ext = os.path.splitext(simFile)
    self.fmuPath = self.name + '.fmu' 
    self.time = -1
    self.dt = dt

    # if slx provided generate FMU
    if ext == '.slx':
      genFmu(simFile,dt)
    
    # initialize
    self.setController(ctrlFkt)
    self.readFmu()
    self.initFmu()
  
  # add the control function to class
  def setController(self,controller):
    self.control = types.MethodType(controller, self)
    if DEBUG: print('Controll function added')
  # initialize FMU
  def initFmu(self):
    if DEBUG: print('Initializing FMU')
    self.fmu.instantiate()
    self.fmu.setupExperiment(startTime=0, tolerance=1e-6, stopTime=self.stopTime)
    self.fmu.enterInitializationMode()
    self.fmu.exitInitializationMode()
    if DEBUG: print('Reading IOs FMU')
    self.getIO()

  # read FMU
  def readFmu(self):
    if DEBUG: print('Starting to read FMU')
    self.description = read_model_description(self.fmuPath)
    self.zipDir = extract(self.fmuPath)
    self.fmu = FMU2Slave(guid=self.description.guid,
                 unzipDirectory=self.zipDir,
                 modelIdentifier=self.description.coSimulation.modelIdentifier,
                 instanceName='instance1')
    if DEBUG: print('Success creating FMU object')
    if LOG: print('Using: {}\nUnzipped in {}'.format(self.fmuPath,self.zipDir))

  # reset FMU to initial state
  def resetFmu(self):
    if DEBUG: print('Initializing FMU')
    self.fmu.reset()
    self.fmu.setupExperiment(startTime=0, tolerance=1e-6, stopTime=self.stopTime)
    self.fmu.enterInitializationMode()
    self.fmu.exitInitializationMode()

  def getIO(self):
    self.input = [x for x in self.description.modelVariables if x.name.startswith(tuple(['I_','Control']))]
    self.output = [x for x in self.description.modelVariables if x.name.startswith(tuple(['O_','Measurement']))]
    self.vrs = [x for x in self.description.modelVariables if not x.name.startswith(tuple(['O_','I_','Control','Measurement']))]

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
  def closeFmu(self):
    if LOG: print('Close fmu and try deleting unzipdir')
    self.fmu.terminate()
    self.fmu.freeInstance()
    shutil.rmtree(self.zipDir, ignore_errors=True)
  
  def simulate(self,stopTime=-1):
    if not stopTime == -1:
      self.stopTime = stopTime

    if LOG: print('###########################################')
    if LOG: print('Starting simulation with timestep: {}'.format(self.dt))
    if self.time > -1:
      self.resetFmu()
    self.time = 0
    self.inputs, result, stepresult = [], [], []

    # main simulation loop
    while self.time <= self.stopTime:
      #set input
      self.control(stepresult)
      #calculate step
      with contextlib.redirect_stdout(None):
        self.fmu.doStep(currentCommunicationPoint=self.time,communicationStepSize=self.dt) 
      if DEBUG: print('Simulation time [s]: {}'.format(self.time))
      #get output
      stepresult = [[self.fmu.getReal([x.valueReference])[0] for x in self.output]]
      result += stepresult
      self.time += self.dt
    if LOG: print('Simulation done.')
    self.outputs = result
    
  # set checkpoint for rollback
  def setCheckpoint(self):
    self.Checkpoint[self.time] = self.fmu.getFMUstate()
  
  # set checkpoint state at specific time
  def rollback(self, time):
    self.fmu.setFMUstate(state = self.Checkpoint[time])
    self.time = time
 
  # write results to pickle dump file
  def exportClass(self):
    output = {'name' : self.name,
              'time' : self.time,
              'startTime' : 0,
              'stopTime' : self.stopTime,
              'Idata' : self.inputs,
              'Odata' : self.outputs,
              'Header' : ['Time'] + self.Inames + self.Onames}

    fn = os.path.join('results', datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S.dump'))
    if not os.path.isdir('results'):
      os.mkdir('results')
    with open(fn,'wb') as f:
      pickle.dump(output,f)
    if LOG: print('Output written to "{}"'.format(fn)) 

# wrapper function for multithread simulation 
def thread_helper (slx,stdout=None,dt=1e-5,stopTime=10,ctrlFkt=default_control.default_controller):
  # make sure FMU file is available at this point
  name, _ = os.path.splitext(slx)
  slx = name + '.fmu' 
  simObj = sim(slx,stdout,dt,stopTime,ctrlFkt)
  simObj.simulate()
  simObj.exportClass()
  sys.stdout = sys.__stdout__

# function to generate FMU  
def genFmu(slx,dt):
  if LOG: print('Creating FMU')
  # posix export currently not working
  if os.name == 'posix':
    os.system('cmd.exe /c wslmatlab.cmd {} {}'.format(slx, dt))
    if DEBUG: print('Created FMU for Linux')
  elif os.name == 'nt':
    from slxToFmu import exportFmu
    exportFmu(slx,dt)
    if DEBUG: print('Created FMU for Windows')
  else:
    print('FMU generation failed. System {} not found'.format(os.name))

def pdPlot(df,field):
  plt.plot(df.Time,df[field])
  plt.title(df[field])
  plt.show()

if __name__ == '__main__':
  simObj = sim('../Matlab/SubmodelFMU/Test/TestFMU.slx')
  simObj.simulate()
  simObj.exportClass()

  simObj = sim('../Matlab/SubmodelFMU/Test/TestFMU.fmu')
  simObj.simulate()
  simObj.exportClass()
  #show results
  import pickle, pathlib
  import matplotlib.pyplot as plt
  import pandas as pd

  for fn in pathlib.Path('results').iterdir():
    print(fn)
    if fn.name.startswith('2022-'):
      with open(fn,'rb') as f:
        simOut = pickle.load(f)
  df = pd.concat([pd.DataFrame(simOut['Idata']),pd.DataFrame(simOut['Odata'])],axis=1)
  df.columns = simOut['Header']

  pdPlot(df, df.columns[6])
  
