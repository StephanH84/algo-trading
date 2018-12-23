# Data:
#12 currency pairs (time series) over roughly 5 years

from common import load_data_structure
import math, datetime

def time_features(dt):
    min_f = math.sin(2*math.pi * dt.minute / 60.)
    hour_f = math.sin(2*math.pi * dt.hour / 24.)
    day_f = math.sin(2*math.pi * dt.weekday() / 7.)
    return min_f, hour_f, day_f

def parse_time(text):
    year = int(text[0:4])
    month = int(text[5:7])
    day = int(text[8:10])

    hour = int(text[11:13])
    min = int(text[14:16])
    sec = int(text[17:19])
    return datetime.datetime(year, month, day, hour, min, sec)

class Data():
    def __init__(self, data_folder):
        self.closing = None
        self.state_space = None
        self.data_folder = data_folder

        self.load()

    def load(self):
        self.closing = load_data_structure(self.data_folder + "EURUSD-closing.json")
        self.state_space = load_data_structure(self.data_folder + "EURUSD-state-space.json")
        self.it = self.iterator()

    def next(self):
        return next(self.it)

    def iterator(self):
        for v in zip(self.closing[8:], self.state_space):
            closing = v[0][1]
            time = time_features(parse_time(v[0]))
            features = v[1][1]

            yield [closing, features, time]
            
    def reset(self):
        self.it = self.iterator()