import pandas as pd
import numpy as np
class preprocess_data:
    def __init__(self) -> None:
        self.data = pd.read_csv('Data/PET_PRI_GND_DCUS_NUS_W.csv')
    def preproces(self):
        #Since the output for this is False, there is no null data
        print(self.data.isnull().sum().any())

p = preprocess_data()
p.preproces()