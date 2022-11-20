from multiprocessing import Pool
import time, os
import fmuSim

DEBUG = os.getenv("DEBUG", None) is not None
LOG = os.getenv("LOG", None) is not None

config = [
  [('../Matlab/SubmodelFMU/Test/TestFMU.slx','write1'), {'stopTime':13,'dt':1e-5}],
  [('../Matlab/SubmodelFMU/Test/TestFMU.fmu','write2'), {'stopTime':10}],
  [('../Matlab/SubmodelFMU/Test/TestFMU.fmu','write3'), {'stopTime':1}],
  [('../Matlab/SubmodelFMU/Test/TestFMU.slx','write1'), {'stopTime':13,'dt':1e-5}],
  [('../Matlab/SubmodelFMU/Test/TestFMU.slx','write2'), {'stopTime':10}],
  [('../Matlab/SubmodelFMU/Test/TestFMU.fmu','write3'), {'stopTime':1}],
  ]

if __name__ == '__main__':
  start = time.time()
  #generating FMU
  slx = config[0][0][0]
  genFmu = config[0][0][1]
  dt = config[0][1]['dt']
  name, ext  = os.path.splitext(slx)
  fmuPath = name + '.fmu' 
  if ext == '.slx':
    fmuSim.genFmu(slx,dt)

  #start async pool
  pool = Pool(processes=3)
  for args, kwds in config:
    pool.apply_async(fmuSim.thread_helper, args=args, kwds=kwds)
  pool.close()
  pool.join()
  end = time.time()

  print('Simulation took {} seconds.'.format(end-start))

