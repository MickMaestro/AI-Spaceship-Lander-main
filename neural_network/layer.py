import numpy as np
np.random.seed(0)
class DenseLayer:
    '''
    Implementation of Dense Layer.
    Constructor: Initialize the weights and bias with shapes according to input shape and hidden neurons
                 Implemented multiple types, default is standard normal distribution multiplied by small number to keep the output in range
                 Also implemented kaiming he, and xavier initialization which works better with relu and tanh.
    Forward pass: recieves inputs, and do a forward pass by applying matrix dot product
    Backward pass: receives an upstream gradient (derror) and multiplies it by the local gradients
                   multiplication shifts
                   derr/dx = w
                   derr/dw = x
                   addition transfers
                   derr/db = sum(derr)
    '''
    
    def __init__(self, input_shape, hidden_neurons, weights_init='default'):
        if weights_init == 'default':
            self.w = 0.10 * np.random.randn(input_shape, hidden_neurons)
        elif weights_init == 'he':
            self.w = np.random.randn(input_shape, hidden_neurons) * np.sqrt(2 / input_shape)
        elif weights_init == 'd_2':
            self.w = np.random.randn(input_shape, hidden_neurons)
        elif weights_init == 'xavier':
            self.w = np.random.randn(input_shape, hidden_neurons) * np.sqrt(2 / (input_shape+hidden_neurons))
        self.b = np.zeros((1,hidden_neurons))
        
    def forward(self, _inputs):
        self.x = _inputs
        self.z = np.dot(_inputs, self.w) + self.b # forward pass
        return self.z
    
    def backward(self, derror):
        self.dx = np.dot(derror, self.w.T) 
        self.dw = np.dot(self.x.T, derror)
        self.db = np.sum(derror, axis=0, keepdims=True)
        return self.dx
