"""
Model Testing Script
Evaluates the trained neural network on test data and calculates RMSE
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import load_data, split_data


def load_trained_weights(weights_path='weights/trained_lr0.8_m0.1_h10.npz'):
    """Load trained weights from NPZ file"""
    data = np.load(weights_path)
    return {
        'w1': data['weights_input_hidden'].astype(float),
        'b1': data['bias_hidden'].astype(float),
        'w2': data['weights_hidden_output'].astype(float),
        'b2': data['bias_output'].astype(float)
    }


def sigmoid(x, lambda_val=0.8):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-lambda_val * x))

def forward_pass(input_data, weights):
    """Perform forward pass through the network"""
    # Hidden layer
    z1 = np.dot(input_data, weights['w1']) + weights['b1']
    a1 = sigmoid(z1, lambda_val=0.8)

    # Output layer
    z2 = np.dot(a1, weights['w2']) + weights['b2']
    a2 = sigmoid(z2, lambda_val=0.8)

    return a2


def calculate_rmse(predictions, targets):
    """Calculate Root Mean Squared Error"""
    mse = np.mean((predictions - targets) ** 2)
    return np.sqrt(mse)


def plot_predictions_vs_actual(predictions, targets, output_dir='results'):
    """
    Plot predicted vs actual values for both X and Y velocities.

    Args:
        predictions: Predicted velocity values (N x 2)
        targets: Actual velocity values (N x 2)
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # X Velocity scatter plot
    ax1.scatter(targets[:, 0], predictions[:, 0], alpha=0.6, s=20, color='darkgreen', edgecolors='k', linewidths=0.5)
    ax1.plot([targets[:, 0].min(), targets[:, 0].max()],
             [targets[:, 0].min(), targets[:, 0].max()],
             color='crimson', linestyle='--', linewidth=2.5, label='Perfect Prediction')
    ax1.set_xlabel('Actual X Velocity', fontsize=12)
    ax1.set_ylabel('Predicted X Velocity', fontsize=12)
    ax1.set_title('X Velocity: Predicted vs Actual', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Y Velocity scatter plot
    ax2.scatter(targets[:, 1], predictions[:, 1], alpha=0.6, s=20, color='darkgreen', edgecolors='k', linewidths=0.5)
    ax2.plot([targets[:, 1].min(), targets[:, 1].max()],
             [targets[:, 1].min(), targets[:, 1].max()],
             color='crimson', linestyle='--', linewidth=2.5, label='Perfect Prediction')
    ax2.set_xlabel('Actual Y Velocity', fontsize=12)
    ax2.set_ylabel('Predicted Y Velocity', fontsize=12)
    ax2.set_title('Y Velocity: Predicted vs Actual', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Model Predictions vs Actual Values', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'predictions_vs_actual.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'[OK] Prediction scatter plots saved to: {filepath}')

    plt.close()


def plot_error_distribution(predictions, targets, output_dir='results'):
    """
    Plot error distribution histograms for both X and Y velocities.

    Args:
        predictions: Predicted velocity values (N x 2)
        targets: Actual velocity values (N x 2)
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    errors_x = predictions[:, 0] - targets[:, 0]
    errors_y = predictions[:, 1] - targets[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # X Velocity error distribution
    ax1.hist(errors_x, bins=50, edgecolor='black', alpha=0.7, color='darkgreen')
    ax1.axvline(x=0, color='crimson', linestyle='--', linewidth=2.5, label='Zero Error')
    ax1.axvline(x=np.mean(errors_x), color='dimgray', linestyle='-', linewidth=2.5,
                label=f'Mean Error: {np.mean(errors_x):.4f}')
    ax1.set_xlabel('Prediction Error', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'X Velocity Error Distribution (Std: {np.std(errors_x):.4f})',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # Y Velocity error distribution
    ax2.hist(errors_y, bins=50, edgecolor='black', alpha=0.7, color='darkgreen')
    ax2.axvline(x=0, color='crimson', linestyle='--', linewidth=2.5, label='Zero Error')
    ax2.axvline(x=np.mean(errors_y), color='dimgray', linestyle='-', linewidth=2.5,
                label=f'Mean Error: {np.mean(errors_y):.4f}')
    ax2.set_xlabel('Prediction Error', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Y Velocity Error Distribution (Std: {np.std(errors_y):.4f})',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Prediction Error Distributions', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'error_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'[OK] Error distribution plots saved to: {filepath}')

    plt.close()


def evaluate_model(data_path='data/data_195k.csv',
                   weights_path='weights/updated_try_1.npz'):
    """
    Evaluate the trained model on test data

    Args:
        data_path: Path to the dataset
        weights_path: Path to trained weights

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print("Model Evaluation on Test Set")
    print(f"{'='*60}")

    # Load data
    print(f"Loading data from: {data_path}")
    df = load_data(data_path)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        df, train_split_ratio=0.7, val_split_ratio=0.15
    )

    print(f"Test set size: {len(X_test)} samples")

    # Load weights
    print(f"Loading weights from: {weights_path}")
    weights = load_trained_weights(weights_path)
    print(f"  W1 shape: {weights['w1'].shape}")
    print(f"  B1 shape: {weights['b1'].shape}")
    print(f"  W2 shape: {weights['w2'].shape}")
    print(f"  B2 shape: {weights['b2'].shape}")

    # Evaluate
    print("\nRunning inference on test set...")
    all_predictions = []
    all_targets = []

    for i in range(len(X_test)):
        input_data = np.expand_dims(X_test[i], axis=0).astype(np.float32)
        target = y_test[i].astype(np.float32)

        prediction = forward_pass(input_data, weights)
        all_predictions.append(prediction.flatten())
        all_targets.append(target)

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate metrics
    overall_rmse = calculate_rmse(all_predictions, all_targets)
    x_vel_rmse = calculate_rmse(all_predictions[:, 0], all_targets[:, 0])
    y_vel_rmse = calculate_rmse(all_predictions[:, 1], all_targets[:, 1])

    # Print results
    print(f"\n{'='*60}")
    print("Test Set Results:")
    print(f"{'='*60}")
    print(f"Overall RMSE:     {overall_rmse:.6f}")
    print(f"X Velocity RMSE:  {x_vel_rmse:.6f}")
    print(f"Y Velocity RMSE:  {y_vel_rmse:.6f}")
    print(f"{'='*60}\n")

    # Sample predictions
    print("Sample Predictions (first 5):")
    print(f"{'Target X':>12} {'Target Y':>12} {'Pred X':>12} {'Pred Y':>12} {'Error':>12}")
    print("-" * 60)
    for i in range(min(5, len(all_targets))):
        error = np.sqrt(np.sum((all_predictions[i] - all_targets[i]) ** 2))
        print(f"{all_targets[i][0]:12.4f} {all_targets[i][1]:12.4f} "
              f"{all_predictions[i][0]:12.4f} {all_predictions[i][1]:12.4f} "
              f"{error:12.4f}")
    print()

    # Generate visualisations in a subfolder based on weights file
    print("Generating visualisations...")
    # Extract weights filename without extension
    weights_name = os.path.splitext(os.path.basename(weights_path))[0]
    output_subdir = os.path.join('results', weights_name)

    plot_predictions_vs_actual(all_predictions, all_targets, output_dir=output_subdir)
    plot_error_distribution(all_predictions, all_targets, output_dir=output_subdir)
    print()

    return {
        'overall_rmse': overall_rmse,
        'x_vel_rmse': x_vel_rmse,
        'y_vel_rmse': y_vel_rmse,
        'predictions': all_predictions,
        'targets': all_targets
    }


if __name__ == "__main__":
    # Evaluate the model
    results = evaluate_model(
        data_path='data/data_195k.csv',
        weights_path='weights/trained_lr0.8_m0.1_h10.npz'
    )

    print("Evaluation complete!")
    print(f"Final Test RMSE: {results['overall_rmse']:.6f}")
