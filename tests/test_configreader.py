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
        assert reader.get(reader.get_sectionnames()[1]) == {'FMU_path': '../../matlab/00-Example/TestGrid.fmu', \
        'start_time': 0, 'stop_time': 20, 'dt': 0.5, 'action_interval': 0.5}

    def test_get_sections(self):
        reader = cfg_reader.configreader(self.test_folder / 'test.cfg')
        assert reader.get_sections(reader.get_sectionnames()) == {'DEFAULT': {}, \
        'FMU': {'FMU_path': '../../matlab/00-Example/TestGrid.fmu', 'start_time': 0, 'stop_time': 20, \
        'dt': 0.5, 'action_interval': 0.5}, \
        'Reinforcement Learning': {'var1': "'test1'", 'var2': True, 'var3': '[1, 3, 5]'}}