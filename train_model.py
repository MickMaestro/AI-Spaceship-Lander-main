"""
Neural Network Training Script
Trains a 2-layer MLP for the lunar lander game
Architecture: 2 inputs -> N hidden -> 2 outputs
Outputs: X_velocity, Y_velocity predictions
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from neural_network.layer import DenseLayer
from neural_network.activation import Sigmoid
from neural_network.loss import MSE, RMSE, Error
from neural_network.optimizer import SgdMomentum
from utils import load_data, split_data

np.random.seed(21)

class NeuralNetwork:
    '''
    Implementation of MLP/Neural Network Architecture.
    '''
    def __init__(self, layers, loss, mse, rmse):
        self.layers = layers
        self.loss_function = loss
        self.mse = mse
        self.rmse = rmse
        self.grad_loss = None

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def forward_pass(self, input_data, target):
        output = self.forward(input_data)
        _, grad_loss = self.loss_function.forward(target, output)
        self.grad_loss = grad_loss
        mse_err = self.mse.forward(target, output)
        rmse_err = self.rmse.forward(target, output)
        return (mse_err, rmse_err), output

    def backward_pass(self):
        self.backward(self.grad_loss)


def stopping_criterion(val_loss):
    '''
    Stop training if RMSE loss plateaus for 3 epochs
    '''
    format_loss = [format(num, '.5f') for num in val_loss]

    if format_loss[0] == format_loss[1] == format_loss[2]:
        return 'stop'
    elif format_loss[0] < format_loss[1] < format_loss[2]:
        return 'stop'
    else:
        return 'continue'


def plot_training_curves(t_loss_mse, v_loss_mse, t_loss_rmse, v_loss_rmse,
                          learning_rate, momentum, hidden_neurons, output_dir='results'):
    """
    Generate and save training curve visualisations.

    Args:
        t_loss_mse: Training MSE loss history
        v_loss_mse: Validation MSE loss history
        t_loss_rmse: Training RMSE loss history
        v_loss_rmse: Validation RMSE loss history
        learning_rate: Learning rate used for training
        momentum: Momentum used for training
        hidden_neurons: Number of hidden neurons
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(t_loss_mse) + 1)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot MSE
    ax1.plot(epochs, t_loss_mse, color='darkgreen', linestyle='-', label='Training MSE', linewidth=2.5)
    ax1.plot(epochs, v_loss_mse, color='crimson', linestyle='--', label='Validation MSE', linewidth=2.5)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Mean Squared Error', fontsize=12)
    ax1.set_title('MSE Loss Over Epochs', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot RMSE
    ax2.plot(epochs, t_loss_rmse, color='darkgreen', linestyle='-', label='Training RMSE', linewidth=2.5)
    ax2.plot(epochs, v_loss_rmse, color='crimson', linestyle='--', label='Validation RMSE', linewidth=2.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Root Mean Squared Error', fontsize=12)
    ax2.set_title('RMSE Loss Over Epochs', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'Training Curves (LR={learning_rate}, Momentum={momentum}, Hidden={hidden_neurons})',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save plot
    filename = f'training_curves_lr{learning_rate}_m{momentum}_h{hidden_neurons}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'\n[OK] Training curves saved to: {filepath}')

    plt.close()


def train_network(data_path='data/data_195k.csv',
                  learning_rate=1.0,
                  momentum=0.5,
                  hidden_neurons=10,
                  epochs=20,
                  save_weights=True):
    """
    Train the neural network and save weights

    Args:
        data_path: Path to training data CSV
        learning_rate: Learning rate for optimizer
        momentum: Momentum for SGD optimizer
        hidden_neurons: Number of neurons in hidden layer
        epochs: Maximum number of training epochs
        save_weights: Whether to save trained weights
    """
    print(f"\n{'='*60}")
    print(f"Training Neural Network")
    print(f"{'='*60}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Momentum: {momentum}")
    print(f"Hidden Neurons: {hidden_neurons}")
    print(f"Max Epochs: {epochs}")
    print(f"{'='*60}\n")

    # Load and split data
    print("Loading data...")
    df = load_data(data_path)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        df, train_split_ratio=0.7, val_split_ratio=0.15
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}\n")

    # Initialize network
    loss = Error()
    mse = MSE()
    rmse = RMSE()
    optim = SgdMomentum(lr=learning_rate, beta=momentum)

    dense_layer = DenseLayer(2, hidden_neurons, weights_init='default')
    activation_hidden = Sigmoid()
    output_layer = DenseLayer(hidden_neurons, 2, weights_init='default')
    activation_output = Sigmoid()

    network = NeuralNetwork(
        [dense_layer, activation_hidden, output_layer, activation_output],
        loss, mse, rmse
    )

    # Training loop
    t_loss_mse = []
    v_loss_mse = []
    t_loss_rmse = []
    v_loss_rmse = []

    print("Starting training...")
    for epoch in range(epochs):
        total_train_loss_mse = 0
        total_val_loss_mse = 0
        total_train_loss_rmse = 0
        total_val_loss_rmse = 0

        # Training
        for i in range(len(X_train)):
            input_data = np.array(np.expand_dims(X_train[i], axis=0), dtype=np.float32)
            target = np.array(y_train[i], dtype=np.float32)

            mse_rmse_loss, _ = network.forward_pass(input_data, target)
            total_train_loss_mse += mse_rmse_loss[0]
            total_train_loss_rmse += mse_rmse_loss[1][0]

            network.backward_pass()
            optim.calculate((dense_layer.dw, output_layer.dw, dense_layer.db, output_layer.db))
            optim.update(vars(dense_layer), vars(output_layer))

        # Validation
        for j in range(len(X_val)):
            val_input_data = np.array(np.expand_dims(X_val[j], axis=0), dtype=np.float32)
            val_target_output = np.array(y_val[j], dtype=np.float32)

            mse_rmse_loss_val, output = network.forward_pass(val_input_data, val_target_output)
            total_val_loss_mse += mse_rmse_loss_val[0]
            total_val_loss_rmse += mse_rmse_loss_val[1][0]

        # Calculate average losses
        t_loss_mse.append(total_train_loss_mse / len(X_train))
        v_loss_mse.append(total_val_loss_mse / len(X_val))
        t_loss_rmse.append(total_train_loss_rmse / len(X_train))
        v_loss_rmse.append(total_val_loss_rmse / len(X_val))

        print(f'Epoch {epoch+1:3d}: '
              f'T_MSE={t_loss_mse[-1]:.6f}, '
              f'V_MSE={v_loss_mse[-1]:.6f}, '
              f'T_RMSE={t_loss_rmse[-1]:.6f}, '
              f'V_RMSE={v_loss_rmse[-1]:.6f}')

        # Check stopping criterion
        if len(v_loss_rmse) >= 3:
            if stopping_criterion(v_loss_rmse[-3:]) == 'stop':
                print('\nStopping criterion met!')
                break

    # Plot training curves
    plot_training_curves(t_loss_mse, v_loss_mse, t_loss_rmse, v_loss_rmse,
                         learning_rate, momentum, hidden_neurons)

    # Save weights
    if save_weights:
        weights_filename = f'weights/trained_lr{learning_rate}_m{momentum}_h{hidden_neurons}.npz'
        np.savez(weights_filename,
                 weights_input_hidden=dense_layer.w,
                 bias_hidden=dense_layer.b,
                 weights_hidden_output=output_layer.w,
                 bias_output=output_layer.b)
        print(f'\n✓ Weights saved to: {weights_filename}')

    # Test set evaluation
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    total_test_loss_rmse = 0
    for i in range(len(X_test)):
        test_input = np.array(np.expand_dims(X_test[i], axis=0), dtype=np.float32)
        test_target = np.array(y_test[i], dtype=np.float32)
        mse_rmse_loss_test, _ = network.forward_pass(test_input, test_target)
        total_test_loss_rmse += mse_rmse_loss_test[1][0]

    test_rmse = total_test_loss_rmse / len(X_test)
    print(f"Final Test RMSE: {test_rmse:.6f}")
    print(f"{'='*60}\n")

    return network, dense_layer, output_layer, test_rmse


if __name__ == "__main__":
    # Train with best hyperparameters found from grid search
    network, dense_layer, output_layer, test_rmse = train_network(
        data_path='data/data_195k.csv',
        learning_rate=0.8,
        momentum=0.1,
        hidden_neurons=10,
        epochs=20,
        save_weights=True
    )
