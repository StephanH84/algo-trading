from selenium import webdriver
from time import sleep
import calendar
import shutil, os
import zipfile
from selenium.webdriver.chrome.options import Options

def fill0(num):
    s = str(num)
    return "0" * (2 - len(s)) + s


exchange_pairs=['AUDJPY', 'AUDNZD', 'AUDUSD', 'CADJPY', 'CHFJPY',
'EURGBP', 'EURJPY', 'EURUSD', 'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDCAD']
years=[str(year) for year in range(2009, 2019)]
months=[[str(calendar.month_name[n]).upper(), fill0(n)] for n in range(1, 13)]

URL0 = "https://truefx.com/dev/data/{YEAR}/{MONTH}-{YEAR}/{PAIR}-{YEAR}-{MON}.zip"
URL1 = "https://truefx.com/dev/data/{YEAR}/{YEAR}-{MON}/{PAIR}-{YEAR}-{MON}.zip"

DIRECTORY = "F:/Dev/Data/truefx/OHLC/"


class Downloader():
    def __init__(self):
        self.it = self.iterator()

    def setup(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        driver = webdriver.Chrome('./chromedriver', chrome_options=chrome_options)

        driver.command_executor()
        chrome_options.add_experimental_option("behavior", "allow")
        chrome_options.add_experimental_option("downloadPath", "/home/stephan/truefx")

        driver.get("https://truefx.com/?page=logina")

        el = driver.find_element_by_name("USERNAME")
        el.send_keys("Stephan")

        el = driver.find_element_by_name("PASSWORD")
        el.send_keys("PUT_PASSWORD")

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
            sleep(0.5)
            if "404" in self.driver.title:
                continue
            else:
                break

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


if __name__ == "__main__":
    downloader = Downloader()
    downloader.setup()

    while True:
        try:
            name = downloader.download()
        except FileExistsError:
            print("File exists.")
        except FileNotFoundError:
            print("Could not be downloaded.")
        except Exception as e:
            print(e)
            break