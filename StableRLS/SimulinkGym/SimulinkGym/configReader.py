# Author Robert Annuth - robert.annuth@tuhh.de
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

def smartParse(obj):
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
