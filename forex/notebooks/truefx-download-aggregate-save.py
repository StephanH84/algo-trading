from selenium import webdriver
from time import sleep
import calendar
import shutil, os

import datetime
from collections import OrderedDict, deque
import math
import numpy as np
import json, codecs
import zipfile

def fill0(num):
    s = str(num)
    return "0" * (2 - len(s)) + s

exchange_pairs=['AUDJPY', 'AUDNZD', 'AUDUSD', 'CADJPY', 'CHFJPY',
'EURGBP', 'EURJPY', 'EURUSD', 'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDCAD']
years=[str(year) for year in range(2009, 2019)]
months=[[str(calendar.month_name[n]).upper(), fill0(n)] for n in range(1, 13)]

URL0 = "https://truefx.com/dev/data/{YEAR}/{MONTH}-{YEAR}/{PAIR}-{YEAR}-{MON}.zip"
URL1 = "https://truefx.com/dev/data/{YEAR}/{YEAR}-{MON}/{PAIR}-{YEAR}-{MON}.zip"

DIRECTORY = "F:/Dev/Data/truefx/"


class Downloader():
    def __init__(self):
        self.it = self.iterator()

    def setup(self):
        driver = webdriver.Chrome("F:\Programme\chromedriver.exe")
        driver.get("https://truefx.com/?page=logina")

        el = driver.find_element_by_name("USERNAME")
        el.send_keys("Stephan")

        el = driver.find_element_by_name("PASSWORD")
        el.send_keys("1M2PBnt")

        el = driver.find_element_by_xpath("//input[@value='Login']")
        el.click()

        self.driver = driver

    def download(self):
        URL_0, URL_1, name = next(self.it)

        filename = DIRECTORY + name
        return_name = name.split(".")[0] + ".json"
        return_file = DIRECTORY + return_name

        if os.path.isfile(return_file):
            raise FileExistsError()

        return_name = name.split(".")[0] + ".csv"
        return_file = DIRECTORY + return_name
        if os.path.isfile(return_file):
            return return_name

        for URL in [URL_0, URL_1]:
            self.driver.get("https://truefx.com/?page=register")
            self.driver.get(URL)
            sleep(1)
            if "404" in self.driver.title:
                continue

        if "404" in self.driver.title:
            raise FileNotFoundError()
        sleep(9)


        try:
            shutil.move("C:/Users/Steph/Downloads/" + name, filename)
        except:
            raise FileNotFoundError()

        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(DIRECTORY)

        os.remove(filename)

        return_name = name.split(".")[0] + ".csv"
        return return_name

    @staticmethod
    def iterator():
        pair = 'EURUSD'
        for year in years:
            for month, mon in months:
                dict_ = {"YEAR": year, "MONTH": month, "PAIR": pair, "MON": mon}
                URL_0 = URL0.format_map(dict_)
                URL_1 = URL1.format_map(dict_)
                name = "{PAIR}-{YEAR}-{MON}.zip".format_map(dict_)

                print(URL_0, URL_1)
                yield URL_0, URL_1, name


def parse_time(text):
    year = int(text[0:4])
    month = int(text[4:6])
    day = int(text[6:8])

    hour = int(text[9:11])
    min = int(text[12:14])
    sec = int(text[15:17])
    return datetime.datetime(year, month, day, hour, min, sec)

def map_datetime(dt):
    dt0 = dt
    return dt0.replace(minute=(dt.minute // 15) * 15, second=0)

def get_ohlc(bucket):
    o, c = bucket[0], bucket[-1]
    h = max(bucket, key=lambda a: a[1])
    l = min(bucket, key=lambda a: a[1])
    return o, h, l, c

def calc_z_scores_parameters(cluster):
    cluster0 = np.asarray(cluster)
    mean = np.mean(cluster0, axis=0)
    variance = np.var(cluster0, axis=0)
    return mean, variance

def z_transform(value, mean, variance):
    result = (np.asarray(value) - mean) / variance
    return result.tolist()

def save_data_structure(structure, file):
    json.dump(structure, codecs.open(file, 'w', encoding='utf-8'), sort_keys=True, indent=4)


def reduce_save(name, directory=DIRECTORY):
    datafile = directory + name
    with open(datafile) as f:
        lines = f.readlines()

    time = []
    price = []
    for line in lines:
        line_split = line.split(",")
        price.append(0.5 * (float(line_split[2]) + float(line_split[3])))  # Ask-Bid-Mitte
        time.append(parse_time(line_split[1]))

    buckets = OrderedDict()
    for t, p in zip(time, price):
        printed_time = str(map_datetime(t))
        if printed_time not in buckets:
            buckets[printed_time] = []

        buckets[printed_time].append((t, p))

    ohlc = OrderedDict()
    for t, bucket in buckets.items():
        ohlc[t] = get_ohlc(bucket)

    closing = list(map(lambda t_v: (t_v[0], t_v[1][3][1]), ohlc.items()))

    save_data_structure(closing, DIRECTORY + name.split(".")[0] + ".json")

    os.remove(datafile)

downloader = Downloader()
downloader.setup()

while True:
    try:
        name = downloader.download()
        reduce_save(name)
    except FileExistsError:
        print("File exists.")
    except FileNotFoundError:
        print("Could not be downloaded.")
    except Exception as e:
        print(e)
        break