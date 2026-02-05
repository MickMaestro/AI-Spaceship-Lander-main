import numpy as np

class MSE:
    '''
    Implementation of MSE Loss.
    Forward pass: recieves target and prediction variable and calculates the mean squared loss
    Backward pass: receives target and prediction, calculates the derivative (should be wary where you put the y_true, y_pred in forward 
                   pass, this determines the negative sign or not) and return the derivative
    '''
    def forward(self, y_true, y_pred):
        self.loss = np.mean((y_true - y_pred)**2)
        return self.loss
    
    def backward(self, y_true, y_pred):
        self.dloss = - 2 * (y_true - y_pred) / len(y_true)
        return self.dloss
    
class MAE:
    def forward(self, y_true, y_pred):
        self.loss = np.mean(y_true - y_pred)
        return self.loss
    
    def backward(self, y_true, y_pred):
        self.dloss = -np.sign(y_true - y_pred) / len(y_true)
        return self.dloss
    
class RMSE:
    '''
    Implementation of RMSE Loss.
    Forward pass: recieves target and prediction variable and calculates the root mean squared loss
    Backward pass: receives target and prediction, calculates the derivative and return the derivative
    '''
    def forward(self, y_true, y_pred):
        self.loss = np.sqrt(np.mean((y_true - y_pred)**2, axis=-1))
        return self.loss
    
    def backward(self, y_true, y_pred):
        denominator = np.sqrt(np.mean((y_true - y_pred)**2, axis=-1))
        self.dloss = - (y_true - y_pred) / (len(y_true) * denominator)
        return self.dloss
    
class Error:
    '''
    Implementation of Error Loss.
    Forward pass: recieves target and prediction variable and simply subtracts the target with prediction
    Backward pass: The derivative would just be negative of forward pass
                   Fun fact: for out problem the derivative of Error and MSE would be same, as the len(y_true) would be 2 which cancels out
                   the constant 2 and this returns the same value.
    '''
    def forward(self, y_true, y_pred):
        self.loss = (y_true - y_pred)
        return np.mean(self.loss), -self.loss
    
    def backward(self):
        self.dloss = - self.loss
        return self.dloss