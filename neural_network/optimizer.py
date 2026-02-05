import numpy as np

class SgdMomentum:
    '''
    Implementation of SGDMomentum.
    Constructor: Initializes learning rate, momentum (beta), and velocities of each layer's weight/bias to zero 
    Calculate: Calculates the velocities of each variable using the formula 
               v = (B * v) - (lr * final_derivative)
    Update: Updates the parameters by adding the velocities to weights 
    '''
    
    def __init__(self, lr=0.8, beta=0.1):
        self.lr = lr
        self.beta = beta
        self.v_w1 = 0
        self.v_w2 = 0
        self.v_b1 = 0
        self.v_b2 = 0
        
    def calculate(self, derror):
        '''
        derror =>
        1st: dense_layer.dw
        2nd: output_layer.dw
        3rd: dense_layer.db
        4th: output_layer.db
        '''
        self.v_w1 = (self.beta * self.v_w1) - (self.lr * derror[0])
        self.v_w2 = (self.beta * self.v_w2) - (self.lr * derror[1])
        self.v_b1 = (self.beta * self.v_b1) - (self.lr * derror[2])
        self.v_b2 = (self.beta * self.v_b2) - (self.lr * derror[3])
    
    def update(self, dense_layer, output_layer):
        # update the weights
        dense_layer['w'] += self.v_w1
        output_layer['w'] += self.v_w2
        dense_layer['b'] += self.v_b1
        output_layer['b'] += self.v_b2