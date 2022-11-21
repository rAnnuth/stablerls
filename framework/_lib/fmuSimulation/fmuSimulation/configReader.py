from fmuSimulation.smartParse import smartParse
from configparser import ConfigParser

class configReader:
    def __init__(self, config):
        parser = ConfigParser()
        parser.optionxform = str #case sensitive
        found = parser.read(config)
        if not found:
            raise ValueError('No config file found!')
        self.parser = parser
    
    def getSections(self):
        return list(dict(self.parser).keys())

    def get(self, section):
        return smartParse(dict(self.parser.items(section)))

    def getMulti(self, sections):
        res = {}
        for section in sections:
            res[section] = smartParse(dict(self.parser.items(section)))
        return res

    def getAgent(self):
        cfg = self.get('Ray')
        sections = self.getSections()
        sections.remove('Ray')
        cfg['env_config'] = self.getMulti(sections)
        return cfg
        

