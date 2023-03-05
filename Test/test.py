#%%
import StableRLS.gymFMU as gymFMU 
import StableRLS.configreader as cfg
import logging
logging.basicConfig(level=logging.DEBUG, filename='test.log', format='%(asctime)s %(levelname)s:%(message)s')

cfg = cfg.configreader('test_files/SimulinkModel.cfg')
fmu = gymFMU.StableRLS(cfg)

# %%
import numpy as np

t = np.arange(0,9,1)

fmu.reset()
terminated = False
while not terminated:
    observation, reward, terminated, truncated, info = fmu.step(np.array([2,3]))
    if fmu.step_count == 10:
        print('Save')
        fmu.save_rollbackstate()
        print(fmu.step_count)
        print(fmu.outputs[fmu.step_count,:])


round(fmu.outputs[-1,0],5)



# %%

fmu.close()

import os
path = os.path.abspath(cfg.__file__)


# %%
import pytest
import numpy as np
from pathlib import Path
import StableRLS.gymFMU as gymFMU 
import StableRLS.configreader as cfg_reader
import StableRLS.createFMU as createFMU

test_folder = Path('.') / 'test_files'
cfg = cfg_reader.configreader(test_folder / 'SimulinkModel.cfg')
createFMU.createFMU(cfg)