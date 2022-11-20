#https://stackoverflow.com/questions/71191926/configparser-read-booleans-string-integer-at-the-same-time-with-python

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