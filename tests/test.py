#%%
import stablerls.gymFMU as gymFMU 
import stablerls.configreader as cfg
import logging
logging.basicConfig(level=logging.DEBUG, filename='test.log', format='%(asctime)s %(levelname)s:%(message)s')

cfg = cfg.configreader('test_files/SimulinkModel.cfg')
fmu = gymFMU.StableRLS(cfg)

# %%
import numpy as np

t = np.arange(0,9,1)

fmu.reset()
terminated = False
truncated = False
while not (terminated or truncated):
    observation, reward, terminated, truncated, info = fmu.step(np.array([2,3]))
    if fmu.step_count == 10:
        print(fmu.step_count)
        print(fmu.outputs[fmu.step_count,:])


round(fmu.outputs[-1,0],5)
fmu.close()



# %%



# %%
import pytest
import numpy as np
from pathlib import Path
import stablerls.gymFMU as gymFMU 
import stablerls.configreader as cfg_reader
import stablerls.createFMU as createFMU

test_folder = Path('.') / 'test_files'
cfg = cfg_reader.configreader(test_folder / 'SimulinkModel.cfg')
createFMU.createFMU(cfg)