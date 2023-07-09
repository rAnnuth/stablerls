# Author Robert Annuth - robert.annuth@tuhh.de
from configparser import ConfigParser


class configreader:
    """Class to read configparser files and get specific sections of the config

    Attributes:
    -----------
    config_name : string
        Path to config that should be read
    """

    def __init__(self, config_name):
        parser = ConfigParser()
        parser.optionxform = str  # case sensitive
        found = parser.read(config_name)
        if not found:
            raise ValueError("No config file found!")
        self.parser = parser

    def get_sectionnames(self):
        """Returns a list of all available sections

        Returns
        -------
        sections : list
            list of available sections
        """
        return list(dict(self.parser).keys())

    # get one specified section from config
    def get(self, section):
        """Returns the parameters of one specific section

        Parameters
        -------
        section : string
            Specified section for returned parameters

        Returns
        -------
        sections : list
            list parameters within the section
        """
        return smart_parse(dict(self.parser.items(section)))

    # get multiple sections from config
    # TODO this should also work for one config
    def get_sections(self, sections):
        """Returns the parameters of multiple sections

        Parameters
        -------
        sections : list
            Specified sections for returned parameters

        Returns
        -------
        sections : dict
            dict of sections with their corresponding parameters
        """
        res = {}
        for section in sections:
            res[section] = smart_parse(dict(self.parser.items(section)))
        return res

    # return the Ray config section with correct dict
    # TODO this can be removed
    def get_ray_agent(self):
        cfg = self.get("Ray")
        sections = self.getSections()
        sections.remove("Ray")
        cfg["env_config"] = self.getMulti(sections)
        return cfg


def smart_parse(obj):
    """Convert the strings of a (nested) dict to the correct datatype.

    Parameters
    -------
    obj : dict
        dictionary containing strings as values

    Returns
    -------
    result : dictionary
        dictionary containing the correct datatype
    """

    if isinstance(obj, dict):
        return {k: smart_parse(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [smart_parse(elem) for elem in obj]

    if isinstance(obj, str):
        if obj == "None":
            return None
        if obj.isnumeric():
            return int(obj)
        if obj.replace(".", "", 1).replace("e", "", 1).replace("-", "").isnumeric():
            return float(obj)
        if obj.upper() in ("TRUE", "FALSE", "T", "F"):
            return obj.upper() in ("TRUE", "T")
    return obj
