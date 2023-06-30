# Tests for StableRLS/gymFMU.py, StableRLS/fmutools and StableRLS/createFMU
import pytest
import numpy as np
from pathlib import Path
import stablerls.gymFMU as gymFMU 
import stablerls.configreader as cfg_reader
import stablerls.createFMU as createFMU


class Test_gymFMU:
    test_folder = Path('.') / 'test_files'
    cfg = cfg_reader.configreader(test_folder / 'SimulinkModel.cfg')
    mdl = test_folder / 'SimulinkModel.slx'

    @pytest.mark.run(order=1)
    def test_createFMU(self):
        createFMU.createFMU(self.cfg, self.mdl )


    @pytest.mark.run(order=2)
    def test_runFMU(self):
        fmu = gymFMU.StableRLS(self.cfg)
        # let the FMU simulate and check the integrated output
        fmu.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            observation, reward, terminated, truncated, info = fmu.step(np.array([2,3]))

        # if this is not 50 something within the simulation went wrong
        assert round(fmu.outputs[-1,0],5) == 50
        fmu.close()

