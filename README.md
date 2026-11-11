# Neural Network from Scratch

An N-layer feed-forward neural network written from scratch in Java, with no ML libraries. It trains with backpropagation and steepest (gradient) descent, and can also run inference on its own. Everything is controlled by a plain-text configuration file.

## Features

- **Any number of layers and nodes.** The architecture comes from the config file, for example `nActivations=2,7,15,8,3` gives 2 inputs, three hidden layers and 3 outputs.
- **Activation functions.** Sigmoid is the default. Hyperbolic tangent (tanh) is also implemented and can be switched in through `activationFunc` / `activationFuncDeriv`.
- **Training and inference are separate.** With `trainMode=true` the network trains and then runs on the test cases. With `trainMode=false` it only runs inference.
- **Input and truth tables.** Inputs and the expected outputs are read from text files (`input_table.txt`, `truth_table.txt`) or filled in manually.
- **Weight initialization.** Weights can be fixed, random within `[minRand, maxRand)`, or loaded from a binary file. Trained weights can be saved to a binary file and reused later.
- **Stopping conditions.** Training stops when the average error drops below `errorThreshold` or after `maxIterations`. It can optionally print keep-alive progress messages.

## How backpropagation works here

For each training case:

1. **Forward pass.** For each hidden and output node, compute the weighted sum θ = Σ a·w and apply the activation function f(θ).
2. **Output error.** For each output, ω = T − a and ψ = ω · f′(θ). The case error is ½ Σ ω².
3. **Backward pass.** Moving backward from the last hidden layer, each node's ω is the sum of ψ (from the layer above) × weight, and its own ψ = ω · f′(θ).
4. **Weight update (steepest descent).** Each weight changes by Δw = λ · a · ψ, where λ is the `learningFactor`.

One iteration goes through every training case. The average error across cases is checked against the stopping conditions.

## Configuration

You pass in a `key=value` text file. If you don't pass one, the program uses `configuration.txt`. The main keys are:

| Key | Purpose |
| --- | --- |
| `nLayers`, `nActivations` | Number of connectivity layers and the nodes per layer |
| `inputFileName`, `truthFileName` | Input table and truth (expected output) table |
| `trainMode` | Train first (`true`), or only run inference (`false`) |
| `typeWeightInit` | `fixed`, `random` or `loaded from file` |
| `weightSaving`, `loadFileName`, `saveFileName` | Save or load weights to/from binary files |
| `learningFactor`, `maxIterations`, `errorThreshold` | Training hyperparameters |
| `minRand`, `maxRand` | Range for random weights |
| `keepAlive` | Print progress every N iterations (`0` turns it off) |

## Running it

Run these from the repository root, since file paths in the configs are relative to it:

```bash
javac NeuralNetwork.java
java NeuralNetwork test_cases/configurationSave.txt
```

This trains a 2-7-15-8-3 network to learn AND, OR and XOR at the same time, then saves the weights to `saved-N-layer.bin`. Example output:

```
Training Exit Information:
Average error was less than the threshold error
Iterations reached: 10964
Average error reached: 1.999903876739077E-4

Truth Table (AND OR XOR):
0.0 0.0 0.0
0.0 1.0 1.0
0.0 1.0 1.0
1.0 1.0 0.0
Outputs:
0.0003   0.0168   0.0159
0.0082   0.9865   0.9851
0.0076   0.9904   0.9895
0.9885   1.0000   0.0142
```

Because the weights start random, the iteration count and exact outputs change from run to run.

Then `java NeuralNetwork test_cases/configurationLoad.txt` loads those saved weights. **Run `configurationSave.txt` first**, because the load config needs the `saved-N-layer.bin` file it creates.

The other configs in `test_cases/` cover single-output XOR (`configurationAB1.txt`), a larger 2-100-5-3 network (`configurationABCD*.txt`). Note that `configurationABC.txt` currently exits with an error, because its `nActivations` lists 4 layer sizes but `nLayers=2` needs 3.

## Image data (not included)

The same network was also trained on image data (images converted to activation arrays). `configuration.txt`, `configuration_test.txt` and `configuration_test_train.txt` are the configs for that. The processed image files (`images*.bin`) and the image-preprocessing code aren't in this repo because of their size, so **those three configs won't run as-is**. The `test_cases/` configs run on their own.
