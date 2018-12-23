import calendar
import shutil, os

import datetime
from collections import OrderedDict, deque
import math
import numpy as np
import json, codecs

def fill0(num):
    s = str(num)
    return "0" * (2 - len(s)) + s

years=[str(year) for year in range(2009, 2019)]
months=[[str(calendar.month_name[n]).upper(), fill0(n)] for n in range(1, 13)]

DIRECTORY = "F:/Dev/Data/truefx/"

def iterator():
    pair = 'EURUSD'
    for year in years:
        for month, mon in months:
            dict_ = {"YEAR": year, "MONTH": month, "PAIR": pair, "MON": mon}
            name = "{PAIR}-{YEAR}-{MON}.json".format_map(dict_)

            yield name

def load_data_structure(file):
    return json.load(codecs.open(file, 'r', encoding='utf-8'))

it = iterator()
closing = []
while True:
    name = next(it)
    filename = DIRECTORY + name
    if not os.path.isfile(filename):
        continue

    print(name)
    closing.extend(load_data_structure(filename))
    print(len(closing))