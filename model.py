from data_preprocessing import preprocess_data
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
class Model:
    def __init__(self) -> None:
        self.learning_rate = 0.001
        self.epochs = 100
        self.sequence_len = 10
        self.hidden_dimension = 2 * self.sequence_len
        self.output_dimension = 1
        self.bptt_truncate = 5
        self.min_clip_value = -10
        self.max_clip_value = 10
        data = preprocess_data('A1')
        self.X_train, self.X_test, self.Y_train, self.Y_test = data.preproces()

        self.U = np.random.uniform(0, 1, (self.hidden_dimension, self.sequence_len))
        self.W = np.random.uniform(0, 1, (self.hidden_dimension, self.hidden_dimension))
        self.V = np.random.uniform(0, 1, (self.output_dimension, self.hidden_dimension))

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def model(self):
        

        for epoch in range(self.epochs):
    # check loss on train
            loss = 0.0
            
            # do a forward pass to get prediction
            for i in range(self.Y_train.shape[0]):
                x, y = self.X_train[i], self.Y_train[i]                    # get input, output values of each record
                prev_s = np.zeros((self.hidden_dimension, 1))   # here, prev-s is the value of the previous activation of hidden layer; which is initialized as all zeroes
                for t in range(self.sequence_len):
                    new_input = np.zeros(x.shape)    # we then do a forward pass for every timestep in the sequence
                    new_input[t] = x[t]              # for this, we define a single input for that timestep
                    mulu = np.dot(self.U, new_input)
                    mulw = np.dot(self.W, prev_s)
                    add = mulw + mulu
                    s = self.sigmoid(add)
                    mulv = np.dot(self.V, s)
                    prev_s = s

            # calculate error 
                loss_per_record = (y - mulv)**2 / 2
                loss += loss_per_record
            loss = loss / float(y.shape[0])

            # check loss on val
            val_loss = 0.0
            for i in range(self.Y_test.shape[0]):
                x, y = self.X_test[i], self.Y_test[i]
                prev_s = np.zeros((self.hidden_dimension, 1))
                for t in range(self.sequence_len):
                    new_input = np.zeros(x.shape)
                    new_input[t] = x[t]
                    mulu = np.dot(self.U, new_input)
                    mulw = np.dot(self.W, prev_s)
                    add = mulw + mulu
                    s = self.sigmoid(add)
                    mulv = np.dot(self.V, s)
                    prev_s = s

                loss_per_record = (y - mulv)**2 / 2
                val_loss += loss_per_record
            val_loss = val_loss / float(y.shape[0])

            print('Epoch: ', epoch + 1, ', Loss: ', loss, ', self.Val Loss: ', val_loss)

            # train model
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
                
                # forward pass
                for t in range(self.sequence_len):
                    new_input = np.zeros(x.shape)
                    new_input[t] = x[t]
                    mulu = np.dot(self.U, new_input)
                    mulw = np.dot(self.W, prev_s)
                    add = mulw + mulu
                    s = self.sigmoid(add)
                    mulv = np.dot(self.V, s)
                    layers.append({'s':s, 'prev_s':prev_s})
                    prev_s = s

                # derivative of pred
                dmulv = (mulv - y)
                
                # backward pass
                for t in range(self.sequence_len):
                    dV_t = np.dot(dmulv, np.transpose(layers[t]['s']))
                    dsv = np.dot(np.transpose(self.V), dmulv)
                    
                    ds = dsv
                    dadd = add * (1 - add) * ds
                    
                    dmulw = dadd * np.ones_like(mulw)

                    dprev_s = np.dot(np.transpose(self.W), dmulw)


                    for i in range(t-1, max(-1, t-self.bptt_truncate-1), -1):
                        ds = dsv + dprev_s
                        dadd = add * (1 - add) * ds

                        dmulw = dadd * np.ones_like(mulw)
                        dmulu = dadd * np.ones_like(mulu)

                        dW_i = np.dot(self.W, layers[t]['prev_s'])
                        dprev_s = np.dot(np.transpose(self.W), dmulw)

                        new_input = np.zeros(x.shape)
                        new_input[t] = x[t]
                        dU_i = np.dot(self.U, new_input)
                        dx = np.dot(np.transpose(self.U), dmulu)

                        dU_t += dU_i
                        dW_t += dW_i
                        
                    dV += dV_t
                    dU += dU_t
                    dW += dW_t
                    if dU.max() > self.max_clip_value:
                        dU[dU > self.max_clip_value] = self.max_clip_value
                    if dV.max() > self.max_clip_value:
                        dV[dV > self.max_clip_value] = self.max_clip_value
                    if dW.max() > self.max_clip_value:
                        dW[dW > self.max_clip_value] = self.max_clip_value
                        
                    
                    if dU.min() < self.min_clip_value:
                        dU[dU < self.min_clip_value] = self.min_clip_value
                    if dV.min() < self.min_clip_value:
                        dV[dV < self.min_clip_value] = self.min_clip_value
                    if dW.min() < self.min_clip_value:
                        dW[dW < self.min_clip_value] = self.min_clip_value
                
                # update
                self.U -= self.learning_rate * dU
                self.V -= self.learning_rate * dV
                self.W -= self.learning_rate * dW

    def predict(self, X,Y):
        preds = []
        for i in range(Y.shape[0]):
            x, y = X[i], Y[i]
            prev_s = np.zeros((self.hidden_dimension, 1))
            # Forward pass
            for t in range(self.sequence_len):
                mulu = np.dot(self.U, x)
                mulw = np.dot(self.W, prev_s)
                add = mulw + mulu
                s = self.sigmoid(add)
                mulv = np.dot(self.V, s)
                prev_s = s

            preds.append(mulv)
        preds = np.array(preds)
        plt.plot(preds[:, 0, 0], 'g')
        plt.plot(Y[:, 0], 'r')
        plt.show()
        return preds, math.sqrt(mean_squared_error(Y[:, 0], preds[:, 0, 0]))
    
    def test_predict(self):
        return self.predict(self.X_test, self.Y_test)
    
    def train_predict(self):
        return self.predict(self.X_train, self.Y_train)