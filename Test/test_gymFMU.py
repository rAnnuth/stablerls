# Tests for StableRLS/gymFMU.py, StableRLS/fmutools and StableRLS/createFMU
#import shutil
import pytest
import numpy as np
from pathlib import Path
import StableRLS.gymFMU as gymFMU 
import StableRLS.configreader as cfg_reader
import StableRLS.createFMU as createFMU


class Test_gymFMU:
    test_folder = Path('.') / 'test_files'
    cfg = cfg_reader.configreader(test_folder / 'SimulinkModel.cfg')

    @pytest.mark.run(order=1)
    def test_createFMU(self):
        createFMU.createFMU(self.cfg)
        self.fmu = gymFMU.StableRLS(cfg)


    @pytest.mark.run(order=2)
    def test_runFMU(self):
        # let the FMU simulate and check the integrated output
        self.fmu.reset()
        terminated = False
        while not terminated:
            observation, reward, terminated, truncated, info = self.fmu.step(np.array([2,3]))

        # if this is not 50 something within the simulation went wrong
        assert round(self.fmu.outputs[-1,0],5) == 50

    @pytest.mark.run(order=3)
    def test_closeFMU(self):
        self.fmu.close()
