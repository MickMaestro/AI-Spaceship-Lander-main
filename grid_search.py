"""
Grid Search for Hyperparameter Optimisation
Systematically evaluates different combinations of learning rate, momentum, and hidden neurons
to identify optimal neural network configuration for the lunar lander task.
"""
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from neural_network.layer import DenseLayer
from neural_network.activation import Sigmoid
from neural_network.loss import MSE, RMSE, Error
from neural_network.optimizer import SgdMomentum
from utils import load_data, split_data


class NeuralNetwork:
    """
    Multilayer Perceptron implementation for grid search.
    Identical to train_model.py but without early stopping.
    """
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


def train_configuration(learning_rate, momentum, hidden_neurons,
                        X_train, y_train, X_val, y_val,
                        epochs=20):
    """
    Train a single configuration and return final metrics.

    Args:
        learning_rate: Learning rate for SGD optimiser
        momentum: Momentum coefficient for SGD
        hidden_neurons: Number of neurons in hidden layer
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Fixed number of training epochs (no early stopping)

    Returns:
        Dictionary with final training and validation metrics
    """
    # Initialise network components
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

    # Training loop (fixed epochs, no early stopping for fair comparison)
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

    # Return final epoch metrics
    return {
        'train_mse_loss': float(total_train_loss_mse / len(X_train)),
        'validation_mse_loss': float(total_val_loss_mse / len(X_val)),
        'train_rmse_loss': float(total_train_loss_rmse / len(X_train)),
        'validation_rmse_loss': float(total_val_loss_rmse / len(X_val))
    }


def grid_search(learning_rates, momentum_values, hidden_neuron_counts,
                data_path='data/data_195k.csv',
                epochs=20,
                output_file='grid_search_results.json'):
    """
    Perform grid search over hyperparameter space.

    Args:
        learning_rates: List of learning rate values to test
        momentum_values: List of momentum values to test
        hidden_neuron_counts: List of hidden neuron counts to test
        data_path: Path to training data CSV
        epochs: Number of epochs per configuration
        output_file: JSON file to save results

    Returns:
        Dictionary containing all results
    """
    print(f"\n{'='*70}")
    print(f"Grid Search: Hyperparameter Optimisation")
    print(f"{'='*70}")
    print(f"Learning rates: {learning_rates}")
    print(f"Momentum values: {momentum_values}")
    print(f"Hidden neurons: {hidden_neuron_counts}")
    print(f"Total configurations: {len(learning_rates) * len(momentum_values) * len(hidden_neuron_counts)}")
    print(f"Epochs per configuration: {epochs}")
    print(f"{'='*70}\n")

    # Load and split data once
    print("Loading data...")
    df = load_data(data_path)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        df, train_split_ratio=0.7, val_split_ratio=0.15
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}\n")

    results = {}
    total_configs = len(learning_rates) * len(momentum_values) * len(hidden_neuron_counts)
    current_config = 0

    # Grid search loop
    for lr in learning_rates:
        for momentum in momentum_values:
            for hidden_neurons in hidden_neuron_counts:
                current_config += 1

                # Create configuration key
                if len(hidden_neuron_counts) == 1:
                    # Format for grid_search_18 style (no hidden neurons in key)
                    key = f"lr_{lr}_momentum_{momentum}"
                else:
                    # Format for grid_search_72 style (includes hidden neurons)
                    key = f"lr_{lr}_momentum_{momentum}_h_neurons_{hidden_neurons}"

                print(f"[{current_config}/{total_configs}] Training {key}...")

                # Train configuration
                metrics = train_configuration(
                    learning_rate=lr,
                    momentum=momentum,
                    hidden_neurons=hidden_neurons,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    epochs=epochs
                )

                # Store results
                results[key] = metrics

                print(f"  Train RMSE: {metrics['train_rmse_loss']:.6f}, "
                      f"Val RMSE: {metrics['validation_rmse_loss']:.6f}\n")

    # Sort results by validation RMSE (best to worst)
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['validation_rmse_loss']))

    # Save sorted results to JSON
    with open(output_file, 'w') as f:
        json.dump(sorted_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Grid search complete! Results saved to: {output_file}")
    print(f"Configurations sorted by validation RMSE (best to worst)")
    print(f"{'='*70}\n")

    # Generate visualisations in a subfolder based on output file
    print("Generating visualisations...")
    # Extract output filename without extension for subfolder name
    output_name = os.path.splitext(os.path.basename(output_file))[0]
    plot_grid_search_results(results, output_name=output_name)
    print()

    return results


def print_best_configurations(results, top_n=5, metric='validation_rmse_loss'):
    """
    Print the top N configurations sorted by specified metric.

    Args:
        results: Dictionary of grid search results
        top_n: Number of top configurations to display
        metric: Metric to sort by (default: validation_rmse_loss)
    """
    # Sort configurations by metric
    sorted_configs = sorted(results.items(), key=lambda x: x[1][metric])

    print(f"\n{'='*70}")
    print(f"Top {top_n} Configurations (sorted by {metric})")
    print(f"{'='*70}")

    for i, (config_name, metrics) in enumerate(sorted_configs[:top_n], 1):
        print(f"\n{i}. {config_name}")
        print(f"   Train MSE:  {metrics['train_mse_loss']:.8f}")
        print(f"   Val MSE:    {metrics['validation_mse_loss']:.8f}")
        print(f"   Train RMSE: {metrics['train_rmse_loss']:.8f}")
        print(f"   Val RMSE:   {metrics['validation_rmse_loss']:.8f}")

    print(f"\n{'='*70}\n")


def plot_grid_search_results(results, output_name='grid_search', output_dir='results'):
    """
    Generate visualisations for grid search results.

    Args:
        results: Dictionary of grid search results
        output_name: Name for the output subfolder (based on JSON filename)
        output_dir: Base directory to save plots
    """
    # Create subfolder for this grid search run
    output_subdir = os.path.join(output_dir, output_name)
    os.makedirs(output_subdir, exist_ok=True)

    # Extract hyperparameters and metrics
    learning_rates = set()
    momentum_values = set()
    hidden_neurons = set()

    for config_name in results.keys():
        parts = config_name.split('_')
        lr_idx = parts.index('lr') + 1
        mom_idx = parts.index('momentum') + 1

        learning_rates.add(float(parts[lr_idx]))
        momentum_values.add(float(parts[mom_idx]))

        if 'neurons' in parts:
            h_idx = parts.index('neurons') + 1
            hidden_neurons.add(int(parts[h_idx]))

    learning_rates = sorted(learning_rates)
    momentum_values = sorted(momentum_values)
    hidden_neurons = sorted(hidden_neurons) if hidden_neurons else [None]

    # Plot 1: Top 10 configurations bar chart
    sorted_configs = sorted(results.items(), key=lambda x: x[1]['validation_rmse_loss'])[:10]
    config_names = [name.replace('_', '\n').replace('lr\n', 'LR=').replace('momentum\n', 'M=').replace('h\nneurons\n', 'H=')
                    for name, _ in sorted_configs]
    val_rmse = [metrics['validation_rmse_loss'] for _, metrics in sorted_configs]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(config_names)), val_rmse, color='darkgreen', edgecolor='black')
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation RMSE', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Configurations by Validation RMSE', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels(config_names, rotation=0, ha='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, val_rmse)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val:.5f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    filepath = os.path.join(output_subdir, 'top10_configurations.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'[OK] Top 10 configurations bar chart saved to: {filepath}')
    plt.close()

    # Plot 2: Heatmap for each hidden neuron count
    for h_neurons in hidden_neurons:
        # Create matrix for heatmap
        heatmap_data = np.zeros((len(momentum_values), len(learning_rates)))

        for i, momentum in enumerate(momentum_values):
            for j, lr in enumerate(learning_rates):
                if h_neurons is not None:
                    key = f"lr_{lr}_momentum_{momentum}_h_neurons_{h_neurons}"
                else:
                    key = f"lr_{lr}_momentum_{momentum}"

                if key in results:
                    heatmap_data[i, j] = results[key]['validation_rmse_loss']
                else:
                    heatmap_data[i, j] = np.nan

        # Make the heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

        # Set ticks and labels
        ax.set_xticks(np.arange(len(learning_rates)))
        ax.set_yticks(np.arange(len(momentum_values)))
        ax.set_xticklabels(learning_rates)
        ax.set_yticklabels(momentum_values)

        ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('Momentum', fontsize=12, fontweight='bold')

        title = f'Validation RMSE Heatmap'
        if h_neurons is not None:
            title += f' (Hidden Neurons = {h_neurons})'
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Validation RMSE', rotation=270, labelpad=20, fontsize=11)

        # Add text annotations
        for i in range(len(momentum_values)):
            for j in range(len(learning_rates)):
                if not np.isnan(heatmap_data[i, j]):
                    text = ax.text(j, i, f'{heatmap_data[i, j]:.4f}',
                                   ha="center", va="center", color="black", fontsize=8)

        plt.tight_layout()
        filename = f'heatmap_h{h_neurons}.png' if h_neurons is not None else 'heatmap.png'
        filepath = os.path.join(output_subdir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f'[OK] Heatmap saved to: {filepath}')
        plt.close()

    # Plot 3: Learning rate vs RMSE for different momentum values
    if len(hidden_neurons) == 1 or None in hidden_neurons:
        h_neurons = hidden_neurons[0] if hidden_neurons[0] is not None else 10

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['darkgreen', 'crimson', 'dimgray']

        for idx, momentum in enumerate(momentum_values):
            rmse_values = []
            for lr in learning_rates:
                if h_neurons is not None:
                    key = f"lr_{lr}_momentum_{momentum}_h_neurons_{h_neurons}"
                else:
                    key = f"lr_{lr}_momentum_{momentum}"

                if key in results:
                    rmse_values.append(results[key]['validation_rmse_loss'])
                else:
                    rmse_values.append(np.nan)

            ax.plot(learning_rates, rmse_values, marker='o', linewidth=2.5,
                    label=f'Momentum = {momentum}', markersize=8,
                    color=colors[idx % len(colors)])

        ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('Validation RMSE', fontsize=12, fontweight='bold')
        title = 'Learning Rate vs Validation RMSE'
        if h_neurons is not None:
            title += f' (Hidden Neurons = {h_neurons})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        plt.tight_layout()
        filepath = os.path.join(output_subdir, 'lr_vs_rmse.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f'[OK] Learning rate vs RMSE plot saved to: {filepath}')
        plt.close()


if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION: Change the SEARCH_MODE value to select which option to run
    # ========================================================================
    # Options: 1, 2, 3, or None
    #   1 = Grid search (18 configs)
    #   2 = Grid search (72 configs)
    #   3 = Grid search (30 configs)
    #   None = Display options without running

    SEARCH_MODE = 2  # Change this to 1, 2, or 3 to run grid search

    # ========================================================================

    if SEARCH_MODE == 1:
        print("\n=== Running Grid Search: Option 1 (18 configs) ===\n")
        results = grid_search(
            learning_rates=[0.001, 0.1, 0.25, 0.5, 0.8, 1.0],
            momentum_values=[0.1, 0.5, 0.9],
            hidden_neuron_counts=[10],
            epochs=20,
            output_file='grid_search_type_1.json'
        )
        print_best_configurations(results)

    elif SEARCH_MODE == 2:
        print("\n=== Running Grid Search: Option 2 (72 configs) ===\n")
        results = grid_search(
            learning_rates=[0.4, 0.5, 0.55, 0.6, 0.67, 0.75],
            momentum_values=[0.1, 0.5, 0.9],
            hidden_neuron_counts=[2, 5, 10, 15],
            epochs=20,
            output_file='grid_search_type_2.json'
        )
        print_best_configurations(results)

    elif SEARCH_MODE == 3:
        print("\n=== Running Grid Search: Option 3 (30 configs) ===\n")
        results = grid_search(
            learning_rates=[0.5, 0.65, 0.75],  # Logarithmic spacing
            momentum_values=[0.45, 0.5, 0.55],  # Include no momentum baseline
            hidden_neuron_counts=[10],  # Sufficient for 2->2 mapping
            epochs=20,
            output_file='grid_search_type_3.json'
        )
        print_best_configurations(results)

    else:
        # Display available options
        print("\n" + "="*70)
        print("Grid Search Configuration")
        print("="*70)
        print("\nNo search mode selected. Set SEARCH_MODE to run grid search:\n")
        print("Option 1: Produce grid_search_type_1.json (18 configs)")
        print("  - Fixed hidden neurons: 10")
        print("  - Learning rates: [0.001, 0.1, 0.25, 0.5, 0.8, 1.0]")
        print("  - Momentum values: [0.1, 0.5, 0.9]")
        print("  - Output: grid_search_type_1.json\n")

        print("Option 2: Produce grid_search_type_2.json (72 configs)")
        print("  - Variable hidden neurons: [2, 5, 10, 15]")
        print("  - Learning rates: [0.001, 0.1, 0.25, 0.5, 0.8, 1.0]")
        print("  - Momentum values: [0.1, 0.5, 0.9]")
        print("  - Output: grid_search_type_2.json\n")

        print("Option 3: Produce grid_search_type_3.json (30 configs)")
        print("  - Hidden neurons: [5, 10]")
        print("  - Learning rates: [0.001, 0.01, 0.1, 0.5, 1.0] (logarithmic)")
        print("  - Momentum values: [0.0, 0.5, 0.9] (includes no momentum)")
        print("  - Output: grid_search_type_3.json (sorted by performance)\n")
        print("To run: Edit this file and change SEARCH_MODE = None to")
        print("SEARCH_MODE = 1, SEARCH_MODE = 2, or SEARCH_MODE = 3")
        print("="*70 + "\n")
