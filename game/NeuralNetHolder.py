import numpy as np
import pandas as pd

class NeuralNetHolder:
    '''
    NeuralNetHolder class implements a multi layer perceptron network for playing the automated lander game.
    Methods:
    init: initializes the weights and biases for the layer and loads the dataframe
    load_weights: load the weights saved in npz file, and assign the values to weights and biases of each layer
    load_df: Read dataframe used for training
    normalize_column: fetches the minimum and maximum values for column X and Y, and apply normalization on input recieved from game
    sigmoid: Implements the logistic function with hyperparameter lambda=0.8
    denormalize_output: fetches the minimum and maximum values for output columns and de-normalize the predictions received from model
    predict: Performs a forward pass and returns the prediction to the Game module
    '''
    def __init__(self):
        super().__init__()
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.df = self.load_df('../data/data_195k.csv')
        self.load_weights('../weights/trained_lr0.8_m0.1_h10.npz')

    def load_weights(self, filename):
        data = np.load(filename) # load weights from npz file
        self.w1 = data['weights_input_hidden'].astype(float)
        self.b1 = data['bias_hidden'].astype(float)
        self.w2 = data['weights_hidden_output'].astype(float)
        self.b2 = data['bias_output'].astype(float)

    def load_df(self, path):
        # load the dataframe
        return pd.read_csv(path, names=['X', 'Y', 'X_Vel', 'Y_Vel'])

    def sigmoid(self, x):
        # implement the sigmoid/logistic function with lambda=0.8; NB: This has to match the sigmoid function used during training
        return 1 / (1 + np.exp(-0.8 * x))

    def normalize_column(self, value):
        # normalise the inputs received from game using training data statistics
        min_val = np.asarray([self.df.X.min(), self.df.Y.min()])
        max_val = np.asarray([self.df.X.max(), self.df.Y.max()])
        return (value - min_val) / (max_val - min_val)


    def denormalize_output(self, normalized_output):
        # denormalise the predictions received after the forward pass
        normalized_output = np.asarray(normalized_output)
        min_vals = np.asarray([self.df.X_Vel.min(), self.df.Y_Vel.min()])
        max_vals = np.asarray([self.df.X_Vel.max(), self.df.Y_Vel.max()])
        return normalized_output * (max_vals - min_vals) + min_vals


    def predict(self, input_row):
        # convert string to floating points
        data = np.array([float(value) for value in input_row.split(',')])
        # normalize the inputs
        input_data = self.normalize_column(data)

        # forward pass - hidden layer
        hidden_output = self.sigmoid(np.dot(input_data, self.w1) + self.b1)
        # forward pass - output layer
        final_output = self.sigmoid(np.dot(hidden_output, self.w2) + self.b2)

        # denormalize outputs
        y_vel, x_vel = self.denormalize_output(final_output.flatten())
        return [x_vel, y_vel]
