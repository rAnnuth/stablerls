# Tests for StableRLS/configreader.py
import shutil
import pytest
from pathlib import Path
import stablerls.configreader as cfg_reader

class Test_configreader:
    test_folder = Path('.') / 'test_files'

    def test_create(self):
        try:
            reader = cfg_reader.configreader(self.test_folder / 'test.cfg')
        except:
            pytest.fail("Unable to create class")  

    def test_get_sectionnames(self):
        reader = cfg_reader.configreader(self.test_folder / 'test.cfg')
        assert reader.get_sectionnames() == ['DEFAULT', 'FMU', 'Reinforcement Learning']

    def test_get(self):
        reader = cfg_reader.configreader(self.test_folder / 'test.cfg')
        assert reader.get(reader.get_sectionnames()[1]) == {'fmuPath': '../../matlab/00-Example/TestGrid.fmu', \
        'startTime': 0, 'stopTime': 20, 'tolerance': 1e-06, 'createFMU': False, 'dt': 0.5}

    def test_get_sections(self):
        reader = cfg_reader.configreader(self.test_folder / 'test.cfg')
        assert reader.get_sections(reader.get_sectionnames()) == {'DEFAULT': {}, \
        'FMU': {'fmuPath': '../../matlab/00-Example/TestGrid.fmu', 'startTime': 0, 'stopTime': 20, \
        'tolerance': 1e-06, 'createFMU': False, 'dt': 0.5}, \
        'Reinforcement Learning': {'actionInterval': 0.5, 'var1': "'test1'", 'var2': '"test2"', 'var3': '[1, 3, 5]'}}