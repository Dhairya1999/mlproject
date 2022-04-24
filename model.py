from data_preprocessing import preprocess_data
import numpy as np
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
class Model:
    def __init__(self) -> None:
        self.learning_rate = 0.001
        self.epochs = 1000
        self.sequence_len = 10
        self.hidden_dimension = 2 * self.sequence_len
        self.output_dimension = 1
        self.bptt_truncate = 5
        data = preprocess_data('A1')
        self.X_train, self.X_test, self.Y_train, self.Y_test = data.preproces()

        self.U = np.random.uniform(0, 1, (self.hidden_dimension, self.sequence_len))
        self.W = np.random.uniform(0, 1, (self.hidden_dimension, self.hidden_dimension))
        self.V = np.random.uniform(0, 1, (self.output_dimension, self.hidden_dimension))

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def model(self):
        for epoch in range(self.epochs):
    
            
            _, train_rmse = self.predict(self.X_train, self.Y_train)
            _, test_rmse = self.predict(self.X_test, self.Y_test)
            print('Epoch: ', epoch + 1, ', Train RMSE: ', train_rmse, ', Test RMSE: ', test_rmse)

            for i in range(self.Y_train.shape[0]):
                x, y = self.X_train[i], self.Y_train[i]
            
                layers = []
                prev_s = np.zeros((self.hidden_dimension, 1))
                dU = np.zeros(self.U.shape)
                dV = np.zeros(self.V.shape)
                dW = np.zeros(self.W.shape)
                
                dU_t = np.zeros(self.U.shape)
                dV_t = np.zeros(self.V.shape)
                dW_t = np.zeros(self.W.shape)
                
                dU_i = np.zeros(self.U.shape)
                dW_i = np.zeros(self.W.shape)
                
                
                for sequence in range(self.sequence_len):
                    new_input = np.zeros(x.shape)
                    new_input[sequence] = x[sequence]
                    mul_w = np.dot(self.W, prev_s)
                    mul_u = np.dot(self.U, new_input)
                    add = mul_w + mul_u
                    s = self.sigmoid(add)
                    mul_v = np.dot(self.V, s)
                    layers.append({'s':s, 'prev_s':prev_s})
                    prev_s = s

                # derivative of pred
                dmul_v = (mul_v - y)
                
                # backward pass
                for t in range(self.sequence_len):
                    dV_t = np.dot(dmul_v, layers[t]['s'].T)
                    dsv = np.dot(self.V.T, dmul_v)
                    
                    d_add = add * (1 - add) * dsv
                    
                    dmul_w = d_add * np.ones_like(mul_w)

                    dprev_s = np.dot(np.transpose(self.W), dmul_w)


                    for i in range(t-1, max(-1, t-self.bptt_truncate-1), -1):
                        ds = dsv + dprev_s
                        d_add = add * (1 - add) * ds

                        dmul_w = d_add * np.ones_like(mul_w)
                        dmul_u = d_add * np.ones_like(mul_u)

                        dW_i = np.dot(self.W, layers[t]['prev_s'])
                        dprev_s = np.dot(np.transpose(self.W), dmul_w)

                        new_input = np.zeros(x.shape)
                        new_input[t] = x[t]
                        dU_i = np.dot(self.U, new_input)
                        dx = np.dot(np.transpose(self.U), dmul_u)

                        dU_t += dU_i
                        dW_t += dW_i
                        
                    dV += dV_t
                    dU += dU_t
                    dW += dW_t

                
                # update
                self.U -= self.learning_rate * dU
                self.V -= self.learning_rate * dV
                self.W -= self.learning_rate * dW

    def predict(self, X,Y, flag=False):
        preds = []
        for i in range(Y.shape[0]):
            x, y = X[i], Y[i]
            prev_s = np.zeros((self.hidden_dimension, 1))
            # Forward pass
            for t in range(self.sequence_len):
                mul_u = np.dot(self.U, x)
                mul_w = np.dot(self.W, prev_s)
                add = mul_w + mul_u
                s = self.sigmoid(add)
                mul_v = np.dot(self.V, s)
                prev_s = s

            preds.append(mul_v)
        preds = np.array(preds)
        if flag:
            plt.plot(preds[:, 0, 0], 'g')
            plt.plot(Y[:, 0], 'r')
            plt.show()
        return preds, math.sqrt(mean_squared_error(Y[:, 0], preds[:, 0, 0]))
    
    def test_predict(self):
        return self.predict(self.X_test, self.Y_test, flag=True)
    
    def train_predict(self):
        return self.predict(self.X_train, self.Y_train, flag=True)