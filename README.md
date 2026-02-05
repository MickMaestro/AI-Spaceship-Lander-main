# AI Spaceship Lander: Neural Network for Autonomous Landing

A lunar lander game featuring an autonomous AI agent trained using a neural network built entirely from scratch. This project was developed as part of the CE889 Neural Networks module, demonstrating supervised learning for continuous control tasks without relying on any machine learning libraries.

## Project Goal

The objective of this project was to implement a neural network from first principles (using only NumPy for matrix operations) that learns to autonomously land a spacecraft on a designated landing zone. The network learns the relationship between the lander's position relative to the target and the optimal velocity required to achieve a successful landing.

**Base Game Credits:** The Pygame-based lunar lander game was developed by **Lewis Veryard** and **Hugo Leon-Garza**. The neural network implementation, training pipeline, hyperparameter optimisation, and AI integration were developed by me as coursework for the CE889 module.

## Demonstration

Watch the trained neural network autonomously land the spacecraft:

<video src="demonstrations/Playthrough_1.mp4" controls width="700"></video>

*The AI agent navigating and landing on the green landing zone*

Additional demonstrations are available in the `demonstrations/` folder:
- `Playthrough_1.mp4` - Successful autonomous landing
- `Playthrough_2.mp4` - More gameplay showing AI decision-making

## Table of Contents

1. [Understanding the Problem](#understanding-the-problem)
2. [Neural Network Architecture](#neural-network-architecture)
3. [The Training Pipeline](#the-training-pipeline)
4. [Hyperparameter Optimisation](#hyperparameter-optimisation)
5. [Results and Analysis](#results-and-analysis)
6. [Project Structure](#project-structure)
7. [Usage Guide](#usage-guide)
8. [Technical Implementation Details](#technical-implementation-details)
   - [Understanding JSON Result Files](#understanding-json-result-files)
   - [Understanding Weight Files](#understanding-weight-files)
9. [Known Limitations](#known-limitations)
10. [Future Improvements](#future-improvements)

---

## Understanding the Problem

The lunar lander task is a continuous control problem where the agent must navigate a spaceship to land safely on the designated green landing zone. The challenge involves:

1. **State Representation:** The lander's position relative to the landing target (X and Y distances)
2. **Action Space:** Thrust (up), rotate left, rotate right
3. **Objective:** Learn optimal velocity predictions that guide the lander to the target

Rather than directly predicting actions, the neural network predicts the **optimal velocity** the lander should have given its current position. The game controller then compares this predicted velocity to the actual velocity and activates the appropriate controls:

- If actual Y velocity > predicted Y velocity → activate thrust (slow descent)
- If actual X velocity < predicted X velocity → rotate right (move right)
- If actual X velocity > predicted X velocity → rotate left (move left)

This approach transforms the reinforcement learning problem into a supervised regression task, where the network learns from human demonstrations.

---

## Neural Network Architecture

### Why a Multilayer Perceptron (MLP)?

For this regression task mapping 2 inputs to 2 outputs, a simple feedforward neural network is enough. The relationship between position and optimal velocity is relatively smooth and can be approximated by a shallow network. Deeper architectures could risk overfitting on a straightforward mapping task.

### Network Structure

```
Input Layer:    2 neurons    [X distance to target, Y distance to target]
                    │
                    ▼
Hidden Layer:   10 neurons   [Sigmoid activation, λ=0.8]
                    │
                    ▼
Output Layer:   2 neurons    [X velocity, Y velocity predictions]
```

### Implementation Details

#### Dense Layer (`neural_network/layer.py`)

The dense layer implements the fundamental operation of neural networks: `z = Wx + b`

```python
def forward(self, _inputs):
    self.x = _inputs
    self.z = np.dot(_inputs, self.w) + self.b
    return self.z

def backward(self, derror):
    self.dx = np.dot(derror, self.w.T)      # Gradient w.r.t. input
    self.dw = np.dot(self.x.T, derror)      # Gradient w.r.t. weights
    self.db = np.sum(derror, axis=0, keepdims=True)  # Gradient w.r.t. bias
    return self.dx
```

**Weight Initialisation Options:**
- **Default:** `0.1 × N(0,1)` - Small random values to prevent saturation
- **He Initialisation:** `N(0,1) × √(2/fan_in)` - Optimal for ReLU activations
- **Xavier Initialisation:** `N(0,1) × √(2/(fan_in + fan_out))` - Optimal for tanh

I use the default initialisation as it works well with sigmoid activations and this small network.

#### Sigmoid Activation (`neural_network/activation.py`)

The sigmoid function introduces non-linearity, allowing the network to learn complex mappings:

```python
σ(x) = 1 / (1 + exp(-λx))
```

I use a lambda parameter (λ=0.8) to control the sigmoid's steepness. This parameter was determined through experimentation and **must match between training and inference**.

**Why Sigmoid over ReLU?**

For this bounded regression task (velocities have physical limits), sigmoid's output range [0,1] naturally constrains predictions. ReLU's unbounded output could potentially make unrealistic velocity predictions during early training.

#### Loss Functions (`neural_network/loss.py`)

I implemented multiple loss functions for different purposes:

- **MSE (Mean Squared Error):** Primary training loss - penalises large errors quadratically
- **RMSE (Root Mean Squared Error):** Evaluation metric - same units as predictions
- **Error:** Raw difference used for gradient computation

The backward pass computes:
```python
∂MSE/∂ŷ = -2(y - ŷ) / n
```

#### SGD with Momentum (`neural_network/optimizer.py`)

Stochastic Gradient Descent with momentum accelerates convergence and dampens oscillations:

```python
v = β × v - lr × gradient
w = w + v
```

Where:
- `lr` = learning rate (controls step size)
- `β` = momentum coefficient (controls influence of previous updates)
- `v` = velocity (accumulated gradient direction)

**Why Momentum?**

Pure SGD can bounce back and forth when navigating steep & narrow valleys in the loss surface. Momentum smooths updates by maintaining a running average of gradients, leading to faster convergence along consistent gradient directions.

---

## The Training Pipeline

### Step 1: Data Collection

The game includes a **Data Collection mode** that records human gameplay:

1. Launch the game and select "Data Collection"
2. Play the game, attempting successful landings
3. Each frame records: `[X_distance, Y_distance, X_velocity, Y_velocity]`
4. Data is saved to `data/ce889_dataCollection.csv`

The dataset used for training is (`data/data_195k.csv`) contains **195,141 samples** of expert gameplay, capturing the relationship between positions and velocities during successful landing attempts.

*I suck at the game and I needed good data*

### Step 2: Data Preprocessing

All features are normalised using **min-max scaling**:

```python
normalised = (value - min) / (max - min)
```

This transforms all values to the [0,1] range, which:
- Prevents features with larger magnitudes from dominating
- Improves gradient flow through sigmoid activations
- Ensures consistent learning across all dimensions

### Step 3: Data Splitting

The dataset is split into three sets:
- **Training (70%):** 136,598 samples - Used for weight updates
- **Validation (15%):** 29,271 samples - Used for hyperparameter tuning and early stopping
- **Test (15%):** 29,272 samples - Final evaluation (never seen during training)

### Step 4: Training Loop

Training uses **online learning** (update weights after each individual sample one at a time) rather than mini batches

```python
for epoch in range(max_epochs):
    for sample in training_data:
        # Forward pass
        prediction = network.forward(sample.input)
        loss = compute_loss(prediction, sample.target)

        # Backward pass
        gradients = network.backward(loss_gradient)

        # Update weights
        optimizer.update(gradients)

    # Check early stopping criterion
    if validation_loss_plateaued():
        break
```

**Early Stopping:** Training stops if validation RMSE doesn't improve for 3 consecutive epochs to prevent overfitting.

### Step 5: Model Evaluation

After training, the model is evaluated on the held out test set:

```bash
python test_model.py
```

This generates:
- Predictions vs Actual scatter plots
- Error distribution histograms
- RMSE metrics for overall and per-output performance

---

## Hyperparameter Optimisation

Finding the right hyperparameters is important for neural network performance. I conducted systematic grid searches to identify optimal values.

### Grid Search Methodology

I evaluated multiple hyperparameter combinations using `grid_search.py`:

| Search Type | Configurations | Parameters Varied |
|-------------|----------------|-------------------|
| Type 1 | 18 configs | LR: [0.001, 0.1, 0.25, 0.5, 0.8, 1.0], M: [0.1, 0.5, 0.9], H: 10 |
| Type 2 | 72 configs | LR: [0.001, 0.1, 0.25, 0.5, 0.8, 1.0], M: [0.1, 0.5, 0.9], H: [2, 5, 10, 15] |
| Type 3 | 30 configs | Refined search around promising regions |

Each configuration trains for a fixed 20 epochs (no early stopping) to ensure a fair comparison.

The grid search script automatically saves all results to a JSON file, sorted from best to worst validation RMSE. This makes it easy to identify top-performing configurations at a glance. The script also generates visualisations for each search run:

- **Top 10 Configurations Bar Chart:** Quick comparison of best hyperparameter combinations
- **RMSE Heatmaps:** Colour-coded matrices showing performance across learning rate and momentum combinations
- **Learning Rate vs RMSE Plots:** Line graphs showing how different momentum values affect learning rate sensitivity

All plots are saved to `results/{search_name}/` with each grid search run getting its own subfolder.

### Results Analysis

The grid search results reveal clear patterns:

**Learning Rate Impact:**
- Very low learning rates (0.001) converge too slowly within 20 epochs
- Learning rates 0.25-1.0 achieve similar final performance
- Higher learning rates (0.8-1.0) converge faster

**Momentum Impact:**
- Low momentum (0.1) provides stable, consistent training
- High momentum (0.9) can cause instability with high learning rates
- The combination of high LR + high momentum leads to overshooting

**Hidden Neurons:**
- 10 neurons are sufficient for this 2→2 mapping task
- Fewer neurons (2, 5) slightly underfit
- More neurons (15) provide no benefit and risk overfitting

### Final Hyperparameter Selection

Based on grid search results and practical testing:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning Rate | 0.8 | Fast convergence without instability |
| Momentum | 0.1 | Stable training, complements high LR |
| Hidden Neurons | 10 | Sufficient capacity without overfitting |
| Lambda (σ) | 0.8 | Balanced sigmoid steepness |

**Why LR=0.8 with Momentum=0.1?**

The heatmap analysis shows that LR=0.8, M=0.1 achieves validation RMSE of 0.0741, ranking among the top configurations. While some high-momentum configurations achieved slightly lower RMSE in grid search, they exhibited higher variance during actual gameplay. The LR=0.8, M=0.1 combination provides the best balance of:
- Fast convergence (fewer epochs needed)
- Training stability (consistent results across runs)
- Robust gameplay performance

---

## Results and Analysis

The training and evaluation scripts automatically generate visualisations saved to the `results/` folder. `train_model.py` produces training curves showing loss over epochs, while `test_model.py` generates prediction scatter plots and error distribution histograms for each model evaluated.

### Training Curves

The training curves for the final model (LR=0.8, M=0.1, H=10) show:

- **Rapid initial descent:** Loss drops significantly in the first 3 epochs
- **Smooth convergence:** No oscillations or instability
- **Minimal overfitting:** Training and validation curves remain close
- **Early stopping:** Training halts around epoch 13 when validation plateaus

### Model Performance

**Test Set RMSE:** ~0.074 (on normalised data [0,1])

**Per-Output Analysis:**
- X Velocity: Standard deviation 0.116, Mean error 0.010
- Y Velocity: Standard deviation 0.053, Mean error 0.007

The Y velocity predictions are more accurate than X velocity, likely because vertical movement (gravity, thrust) follows more predictable physics than horizontal manoeuvring.

### Prediction Quality

The predictions vs actual scatter plots show:
- Strong correlation along the diagonal (accurate predictions)
- Tighter clustering for Y velocity than X velocity
- Some spread at extreme values (edge cases in training data)

The error distributions are approximately normal and centred near zero, indicating unbiased predictions.

---

## Project Structure

```
AI-Spaceship-Lander-main/
├── game/                           # Pygame lunar lander application
│   ├── Main.py                     # Entry point
│   ├── GameLoop.py                 # Main game loop and state management
│   ├── NeuralNetHolder.py          # AI integration - loads weights and runs inference
│   ├── Lander.py                   # Spacecraft physics and collision
│   ├── Surface.py                  # Terrain and landing zone generation
│   ├── DataCollection.py           # Records gameplay for training data
│   ├── Controller.py               # Input abstraction layer
│   ├── EventHandler.py             # Keyboard/mouse input processing
│   ├── Vector.py                   # 2D vector mathematics
│   ├── CollisionUtility.py         # Line intersection collision detection
│   ├── MainMenu.py                 # Main menu UI
│   ├── ResultMenu.py               # Win/lose screen UI
│   ├── GameLogic.py                # Game state updates
│   ├── Files/Config.con            # Game configuration
│   └── Sprites/                    # Game graphics
│
├── neural_network/                 # Custom MLP implementation (from scratch)
│   ├── layer.py                    # Dense layer with forward/backward pass
│   ├── activation.py               # Sigmoid activation function
│   ├── loss.py                     # MSE, RMSE, MAE loss functions
│   └── optimizer.py                # SGD with momentum optimiser
│
├── data/                           # Training datasets
│   ├── data_195k.csv               # 195,141 gameplay samples
│   └── ce889_dataCollection.csv    # Your collected data (if any)
│
├── weights/                        # Trained model weights
│   ├── trained_lr0.8_m0.1_h10.npz  # Primary model (recommended)
│   └── trained_lr0.5_m0.9_h10.npz  # Alternative configuration
│
├── results/                        # Generated visualisations
│   ├── training_curves_*.png       # Loss curves per training run
│   ├── trained_*/                  # Per-model evaluation results
│   │   ├── predictions_vs_actual.png
│   │   └── error_distribution.png
│   └── grid_search_*/              # Per-search visualisations
│       ├── top10_configurations.png
│       ├── heatmap*.png
│       └── lr_vs_rmse.png
│
├── demonstrations/                 # Video demonstrations of AI gameplay
│   ├── Playthrough_1.mp4
│   └── Playthrough_2.mp4
│
├── train_model.py                  # Neural network training script
├── test_model.py                   # Model evaluation and visualisation
├── grid_search.py                  # Hyperparameter optimisation
├── utils.py                        # Data loading and preprocessing
├── grid_search*.json               # Grid search results
└── README.md                       # This file
```

---

## Usage Guide

### Prerequisites

```bash
pip install pygame numpy pandas matplotlib
```

Python 3.7 or higher recommended.

### Running the Game

```bash
cd game
python Main.py
```

**Game Modes:**
1. **Play Game** - Manual control (Arrow keys: Up=thrust, Left/Right=rotate)
2. **Data Collection** - Manual play while recording training data
3. **Neural Network** - Watch the AI play autonomously
4. **Quit** - Exit application

### Training a New Model

```bash
python train_model.py
```

Edit the script to modify hyperparameters:
```python
network, dense_layer, output_layer, test_rmse = train_network(
    data_path='data/data_195k.csv',
    learning_rate=0.8,      # Adjust learning rate
    momentum=0.1,           # Adjust momentum
    hidden_neurons=10,      # Adjust network capacity
    epochs=20,              # Maximum training epochs
    save_weights=True
)
```

**Outputs:**
- Weights: `weights/trained_lr{lr}_m{momentum}_h{hidden}.npz`
- Training curves: `results/training_curves_lr{lr}_m{momentum}_h{hidden}.png`

### Evaluating a Model

```bash
python test_model.py
```

Edit `test_model.py` to change which weights file to evaluate:
```python
weights_filename = 'weights/trained_lr0.8_m0.1_h10.npz'
```

**Outputs:**
- Console metrics (RMSE overall and per-output)
- `results/{weights_name}/predictions_vs_actual.png`
- `results/{weights_name}/error_distribution.png`

### Running Grid Search

```bash
python grid_search.py
```

Configure the search by editing `SEARCH_MODE` in the script:
```python
SEARCH_MODE = 1  # Options: 1, 2, 3, or None
```

**Outputs:**
- JSON results file with all metrics (automatically sorted best to worst by validation RMSE)
- Visualisations in `results/{search_name}/` (heatmaps, top 10 bar chart, LR vs RMSE plots)

### Changing the AI Model in Game

Edit `game/NeuralNetHolder.py` line 23:
```python
self.load_weights('../weights/trained_lr0.8_m0.1_h10.npz')
```

### Game Configuration

Edit `game/Files/Config.con`:
```
FULLSCREEN = TRUE          # TRUE for fullscreen, FALSE for windowed
SCREEN_WIDTH = 1600        # Width in windowed mode
SCREEN_HEIGHT = 1000       # Height in windowed mode
ALL_DATA = FALSE           # FALSE for minimal data collection
```

---

## Technical Implementation Details

### Understanding JSON Result Files

The `grid_search*.json` files store the results of hyperparameter searches. Each file contains metrics for every configuration tested, automatically sorted from best to worst validation RMSE.

**Structure:**
```json
{
  "lr_0.8_momentum_0.1": {
    "train_mse_loss": 0.00827,
    "validation_mse_loss": 0.00817,
    "train_rmse_loss": 0.07430,
    "validation_rmse_loss": 0.07411
  },
  ...
}
```

**How to use them:**

1. **Find optimal hyperparameters:** Open the JSON file - the first entry is the best-performing configuration
2. **Compare configurations:** Look at how different learning rates and momentum values affected performance
3. **Identify patterns:** Use the data to understand which hyperparameter ranges work well for this task
4. **Train final model:** Take the best configuration from grid search and use those values in `train_model.py` to train your final model with early stopping enabled

The JSON files serve as a record of experiments, making it easy to revisit results without re-running lengthy grid searches.

### Understanding Weight Files

When you train a neural network, the "learning" is stored in the **weight matrices** and **bias vectors**. These are the numerical values that get adjusted during training to minimise prediction error.

The weights are saved as `.npz` files (NumPy's compressed archive format) in the `weights/` folder. Each file contains four arrays:

| Array | Shape | Description |
|-------|-------|-------------|
| `weights_input_hidden` (W₁) | 2 × 10 | Connections from 2 input neurons to 10 hidden neurons |
| `bias_hidden` (b₁) | 1 × 10 | Bias for each hidden neuron |
| `weights_hidden_output` (W₂) | 10 × 2 | Connections from 10 hidden neurons to 2 output neurons |
| `bias_output` (b₂) | 1 × 2 | Bias for each output neuron |

The filename encodes the hyperparameters used: `trained_lr0.8_m0.1_h10.npz` means learning rate 0.8, momentum 0.1, and 10 hidden neurons.

**Using Different Weights:**

You can train multiple models with different hyperparameters and compare their performance. To switch which weights the game uses:

1. Train a new model or use an existing weights file
2. Edit `game/NeuralNetHolder.py` line 23 to point to your chosen weights file
3. Run the game in Neural Network mode to test

**Important:** The number of hidden neurons in the weights file must match the network architecture in `NeuralNetHolder.py`. The default is 10 hidden neurons.

### Critical Configuration: Lambda Parameter

The sigmoid activation uses λ=0.8. This value **must be identical** in:
- `neural_network/activation.py` (training)
- `game/NeuralNetHolder.py` (inference)

Mismatched lambda values will cause incorrect predictions.

### Normalisation Consistency

The game loads training data statistics to normalise inputs and denormalise outputs. The same min/max values used during training must be used during inference, which is why `NeuralNetHolder` loads the original training CSV.

### Random Initialisation

Training involves random weight initialisation. To reproduce exact results:
```python
np.random.seed(42)  # Add at script start
```

Without a fixed seed, training runs will produce slightly different weights.

### DPI Awareness

I made a small improvement to the game to include Windows DPI awareness handling to prevent scaling issues on high-resolution displays. I along with some classmates had issues with the game not always being scaled to screens properly.

---

## Known Limitations

Through testing, I observed that the trained model:

- **Performs well** when the landing zone spawns in the **centre region** of the screen
- **Struggles** when the landing zone appears on the **far left or far right edges**

This limitation likely stems from the training data distribution. Human players naturally spend more time navigating towards centrally-located targets, resulting in fewer training samples for extreme edge positions. The model therefore has less experience with the aggressive manoeuvres required when the lander spawns far from an edge-positioned target.

---

## Future Improvements

Several enhancements could address current limitations and improve performance:

1. **Data Augmentation:** Mirror training samples horizontally to balance left/right edge cases and improve generalisation to all landing zone positions.

2. **Curriculum Learning:** Gradually increase task difficulty during training, starting with central landing zones and progressively including edge cases.

3. **Additional Input Features:** Include current velocity, angle, or distance to surface as inputs, giving the network more context for decision-making.

4. **Deeper Architecture:** Experiment with additional hidden layers for potentially more nuanced velocity predictions, though this risks overfitting on the current dataset size.

5. **Alternative Activations:** Test ReLU or Leaky ReLU with proper output scaling, which may improve gradient flow during training.

6. **Batch Normalisation:** Add normalisation between layers to stabilise training and potentially allow higher learning rates.

7. **Reinforcement Learning:** Replace supervised learning with policy gradient methods (e.g., REINFORCE) to learn directly from gameplay rewards rather than human demonstrations.

---

## Acknowledgements

- **Base Game:** Lewis Veryard and Hugo Leon-Garza
- **Module:** CE889 Neural Networks
- **Collision Detection:** Adapted from [Pygame Wiki](https://www.pygame.org/wiki/IntersectingLineDetection)

---

## License

This project was developed for educational purposes as part of my coursework @ The University of Essex
