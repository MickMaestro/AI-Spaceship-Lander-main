import numpy as np

class Sigmoid:
    '''
    Implementation of logistic/Sigmoid function.
    Constructor: Initialize the delta value (lambda), it allows to extend the range of logisitic function
    Two methods 
    Forward pass: recieves inputs and apply it in the formula
    Backward pass: receives an upstream gradient and multiply by sigmoid * (1 - sigmoid)
    '''
    def __init__(self, lmbda=0.8):
        self.lmbda = lmbda # setting the default lmbda value to 0.8
        
    def forward(self, _inputs):
        self.gz = 1 / (1+np.exp(self.lmbda * -_inputs)) # logistic function
        return self.gz
        
    def backward(self, derror):
        self.dgz = derror * self.lmbda * self.gz * (1 - self.gz) # upstream gradient * local gradient of sigmoid
        return self.dgz