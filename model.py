from data_preprocessing import preprocess_data
import numpy as np
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
class Model:
    def __init__(self):
        self.learning_rate = 0.001
        self.epochs = 300
        self.sequence_len = 10
        self.hidden_dimension = 2 * self.sequence_len
        self.output_dimension = 1
        self.bptt_t = 5
        data = preprocess_data('A1')
        self.X_train, self.X_test, self.Y_train, self.Y_test = data.preproces()

        self.Wxh = np.random.uniform(0, 1, (self.hidden_dimension, self.sequence_len))
        self.F = np.random.uniform(0, 1, (self.hidden_dimension, self.hidden_dimension))
        self.Why = np.random.uniform(0, 1, (self.output_dimension, self.hidden_dimension))


    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def model(self):

        for epoch in range(self.epochs):
    
            
            _, train_rmse = self.predict(self.X_train, self.Y_train)
            _, test_rmse = self.predict(self.X_test, self.Y_test)
            print(f'Epoch: {epoch + 1}  Train RMSE:  {train_rmse} Test RMSE:  {test_rmse}')

            for record in range(self.Y_train.shape[0]):
                x, y = self.X_train[record], self.Y_train[record]
            
                #Initialize parameters
                layers = []
                pre_state = np.zeros((self.hidden_dimension, 1))
                dWxh = np.zeros(self.Wxh.shape)
                dWhy = np.zeros(self.Why.shape)
                dF = np.zeros(self.F.shape)
                
                dWxh_t = np.zeros(self.Wxh.shape)
                dWhy_t = np.zeros(self.Why.shape)
                dF_t = np.zeros(self.F.shape)
                
                dWxh_i = np.zeros(self.Wxh.shape)
                dF_i = np.zeros(self.F.shape)
                
                
                for sequence in range(self.sequence_len):
                    temp = np.zeros(x.shape)
                    temp[sequence] = x[sequence]
                    prod_f = np.dot(self.F, pre_state)
                    prod_Wxh = np.dot(self.Wxh, temp)
                    sum1 = prod_f + prod_Wxh
                    state = self.sigmoid(sum1)
                    prod_Why = np.dot(self.Why, state)
                    layers.append({'state':state, 'previous_state':pre_state})
                    pre_state = state

                
                dprod_Why = (prod_Why - y)
                
                
                for num_sequence in range(self.sequence_len):
                    dWhy_t = np.dot(dprod_Why, layers[num_sequence]['state'].T)
                    ds_Why = np.dot(self.Why.T, dprod_Why)
                    
                    d_sum1 = sum1 * (1 - sum1) * ds_Why
                    
                    dprod_f = d_sum1 * np.ones_like(prod_f)

                    dpre_state = np.dot(self.F.T, dprod_f)


                    for _ in range(num_sequence-1, max(-1, num_sequence-self.bptt_t-1), -1):
                        ds = ds_Why + dpre_state
                        d_sum1 = sum1 * (1 - sum1) * ds

                        dprod_f = d_sum1 * np.ones_like(prod_f)
                        

                        dF_i = np.dot(self.F, layers[num_sequence]['previous_state'])
                        dpre_state = np.dot(self.F.T, dprod_f)

                        temp = np.zeros(x.shape)
                        temp[num_sequence] = x[num_sequence]
                        dWxh_i = np.dot(self.Wxh, temp)
                        

                        dWxh_t += dWxh_i
                        dF_t += dF_i
                        
                    dWhy += dWhy_t
                    dWxh += dWxh_t
                    dF += dF_t

                
                
                self.Wxh -= self.learning_rate * dWxh
                self.Why -= self.learning_rate * dWhy
                self.F -= self.learning_rate * dF

    def predict(self, X,Y, flag=False):

        predictions = []
        for record in range(Y.shape[0]):
            pre_state = np.zeros((self.hidden_dimension, 1))
            
            for _ in range(self.sequence_len):
                prod_Wxh = np.dot(self.Wxh, X[record])
                prod_f = np.dot(self.F, pre_state)
                sum1 = prod_f + prod_Wxh
                s = self.sigmoid(sum1)
                prod_Why = np.dot(self.Why, s)
                pre_state = s

            predictions.append(prod_Why)
        predictions = np.array(predictions)
        Y_values = Y[:, 0]
        prediction_values = predictions[:, 0, 0]
        if flag:

            fig = plt.figure(figsize=(14,8))
            plt.plot(prediction_values, 'b')
            plt.plot(Y_values, 'y')

            plt.show()


        return predictions, math.sqrt(mean_squared_error(Y_values, prediction_values))
    
    def test_predict(self):
        return self.predict(self.X_test, self.Y_test, flag=True)
    
    def train_predict(self):
        return self.predict(self.X_train, self.Y_train, flag=True)