import json, codecs

def load_data_structure(file):
    return json.load(codecs.open(file, 'r', encoding='utf-8'))