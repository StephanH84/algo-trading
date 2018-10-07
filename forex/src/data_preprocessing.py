# Data:
#12 currency pairs (time series) over roughly 5 years

class Data():
    def __init__(self, data_folder):
        self.seq = None #list of pairs with timestamp and 198 - 3 (recent one-hot encoded action) dim. vectors

    def preprocess(self):
        pass

    def generate_state_space_data(self):
        pass