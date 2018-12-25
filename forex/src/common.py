import json, codecs
import numpy as np


def hot_encoding(a):
    a_ = np.zeros(3, dtype=np.float32)
    a_[a + 1] = 1.
    return a_


def load_data_structure(file):
    return json.load(codecs.open(file, 'r', encoding='utf-8'))