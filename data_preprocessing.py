import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
class preprocess_data:
    def __init__(self, fuel_grade):
        self.data = pd.read_csv('https://personal.utdallas.edu/~sxs200389/PET_PRI_GND_DCUS_NUS_W.csv')
        self.data = np.array(self.data[fuel_grade])

    def preproces(self):
        #Since the output for this is False, there is no null data
        #print(self.data.isnull().sum().any())
        X_train = []
        Y_train = []

        sequence_len = 10
        num_records = len(self.data) - int(0.2 * len(self.data))
        scaler = MinMaxScaler(feature_range=(0,1))
        for record in range(num_records - sequence_len):
            X_train.append(self.data[record:record+sequence_len])
            Y_train.append(self.data[record+sequence_len])
            
        X_train = np.array(X_train)
        X_train = X_train.reshape(-1,sequence_len)
        X_train = scaler.fit_transform(X_train)
        X_train = np.expand_dims(X_train, axis=2)

        Y_train = np.array(Y_train)
        Y_train = Y_train.reshape(-1,1)
        Y_train = scaler.fit_transform(Y_train)
        Y_train = np.expand_dims(Y_train, axis=1)
        
        X_test = []
        Y_test = []

        for record in range(len(self.data) - num_records, len(self.data)-sequence_len):
            X_test.append(self.data[record:record+sequence_len])
            Y_test.append(self.data[record+sequence_len])

        scaler = MinMaxScaler(feature_range=(0,1))

        X_test = np.array(X_test)
        X_test = X_test.reshape(-1,sequence_len)
        X_test = scaler.fit_transform(X_test)
        X_test = np.expand_dims(X_test, axis=2)

        Y_test = np.array(Y_test)
        Y_test = Y_test.reshape(-1,1)
        Y_test = scaler.fit_transform(Y_test)
        Y_test = np.expand_dims(Y_test, axis=1)

        return X_train, X_test, Y_train, Y_test

