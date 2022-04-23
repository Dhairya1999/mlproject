import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
class preprocess_data:
    def __init__(self) -> None:
        self.data = pd.read_csv('Data/PET_PRI_GND_DCUS_NUS_W.csv')
    def preproces(self, fuel_grade):
        #Since the output for this is False, there is no null data
        print(self.data.isnull().sum().any())
        X, Y = [], []
        for i in range(len(self.data[fuel_grade])-1):
            X.append(self.data[fuel_grade][i])
            Y.append(self.data[fuel_grade][i+1])
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(np.array(X).reshape(-1,1)), scaler.fit_transform(np.array(Y).reshape(-1,1))
p = preprocess_data()
print(p.preproces(fuel_grade='A2'))