# Author Robert Annuth - robert.annuth@tuhh.de
from configparser import ConfigParser


class configReader:
    """
    class to read configparser files and get specific sections of the config
    """

    def __init__(self, config):
        parser = ConfigParser()
        parser.optionxform = str  # case sensitive
        found = parser.read(config)
        if not found:
            raise ValueError('No config file found!')
        self.parser = parser

    # return list of available sections
    def getSections(self):
        return list(dict(self.parser).keys())

    # get one specified section from config
    def get(self, section):
        return smartParse(dict(self.parser.items(section)))

    # get multiple sections from config
    # TODO this should also work for one config
    def getMulti(self, sections):
        res = {}
        for section in sections:
            res[section] = smartParse(dict(self.parser.items(section)))
        return res

    # return the Ray config section with correct dict
    # TODO this can be removed
    def getAgent(self):
        cfg = self.get('Ray')
        sections = self.getSections()
        sections.remove('Ray')
        cfg['env_config'] = self.getMulti(sections)
        return cfg


def smartParse(obj):
    """
    convert the strings of a (nested) dict to the correct datatype.

    Parameters:
        obj (dict): dictionary containing strings as values

    Returns:
        dict: dictionary containing the correct datatype
    """

    if isinstance(obj, dict):
        return {k: smartParse(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [smartParse(elem) for elem in obj]

    if isinstance(obj, str):
        if obj == 'None':
            return None
        if obj.isnumeric():
            return int(obj)
        if obj.replace('.', '', 1).replace('e', '', 1).replace('-', '').isnumeric():
            return float(obj)
        if obj.upper() in ('TRUE', 'FALSE', 'T', 'F'):
            return obj.upper() in ('TRUE', 'T')
    return obj
