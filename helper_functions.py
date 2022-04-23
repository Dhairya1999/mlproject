import numpy as np
class helper_functions:
    def __init__(self):
        pass
    
    def tanh(self, x):
        num1 = np.exp(x)
        num2 = np.exp(-x)
        return (num1 - num2)/(num1 + num2)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def dtanh(self, x):
        return 1 - self.tanh(x) ** 2

    def dsigmoid(self, x):
        temp = self.sigmoid(x)
        return temp * (1 - temp)