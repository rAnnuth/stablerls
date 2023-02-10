# Author Robert Annuth - robert.annuth@tuhh.de
from configparser import ConfigParser


class configreader:
    """
    class to read configparser files and get specific sections of the config

    Attributes:
    -------
        None


    Methods
    -------
    __init__(config_name):
        Reads config specified by config_name.
    get_sectionnames():
        Returns all section names of config.
    get(section):
        Get one specific section.
    get_sections(sections):
        Get multiple sections specified as list.

    #get_ray_agent(self):
    """

    def __init__(self, config_name):
        parser = ConfigParser()
        parser.optionxform = str  # case sensitive
        found = parser.read(config_name)
        if not found:
            raise ValueError('No config file found!')
        self.parser = parser

    # return list of available sections
    def get_sectionnames(self):
        return list(dict(self.parser).keys())

    # get one specified section from config
    def get(self, section):
        return smart_parse(dict(self.parser.items(section)))

    # get multiple sections from config
    # TODO this should also work for one config
    def get_sections(self, sections):
        res = {}
        for section in sections:
            res[section] = smart_parse(dict(self.parser.items(section)))
        return res

    # return the Ray config section with correct dict
    # TODO this can be removed
    def get_ray_agent(self):
        cfg = self.get('Ray')
        sections = self.getSections()
        sections.remove('Ray')
        cfg['env_config'] = self.getMulti(sections)
        return cfg


def smart_parse(obj):
    """
    convert the strings of a (nested) dict to the correct datatype.

    Parameters:
        obj (dict): dictionary containing strings as values

    Returns:
        dict: dictionary containing the correct datatype
    """

    if isinstance(obj, dict):
        return {k: smart_parse(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [smart_parse(elem) for elem in obj]

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
