use serde::{Deserialize, Serialize};
use rand::Rng;

#[derive(Clone, Serialize, Deserialize)]
pub struct Layer {
    pub weights: Vec<Vec<f64>>, // [output_neurons][input_neurons]
    pub biases: Vec<f64>,       // [output_neurons]
}

#[derive(Clone, Serialize, Deserialize)]
pub struct NeuralNet {
    pub layers: Vec<Layer>,
    pub learning_rate: f64,
}

fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(y: f64) -> f64 {
    y * (1.0 - y)
}

impl NeuralNet {
    pub fn new(layer_sizes: &[usize], learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let mut layers = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let in_size = layer_sizes[i];
            let out_size = layer_sizes[i + 1];

            // Xavier/Glorot initialization
            let bound = (6.0 / (in_size + out_size) as f64).sqrt();

            let weights: Vec<Vec<f64>> = (0..out_size)
                .map(|_| {
                    (0..in_size)
                        .map(|_| rng.gen_range(-bound..bound))
                        .collect()
                })
                .collect();

            let biases: Vec<f64> = (0..out_size).map(|_| 0.0).collect();
            layers.push(Layer { weights, biases });
        }

        NeuralNet {
            layers,
            learning_rate,
        }
    }

    /// Forward pass — returns (final_output, all_activations_for_backprop).
    /// activations[0] = input, activations[i] = output of layer i-1.
    pub fn forward(&self, input: &[f64]) -> (f64, Vec<Vec<f64>>) {
        let mut activations: Vec<Vec<f64>> = Vec::with_capacity(self.layers.len() + 1);
        activations.push(input.to_vec());

        let mut current = input.to_vec();

        for (i, layer) in self.layers.iter().enumerate() {
            let mut next = Vec::with_capacity(layer.weights.len());
            for (out_idx, weights_row) in layer.weights.iter().enumerate() {
                let mut sum = layer.biases[out_idx];
                for (in_idx, &w) in weights_row.iter().enumerate() {
                    sum += w * current[in_idx];
                }
                let is_output = i == self.layers.len() - 1;
                next.push(if is_output {
                    sigmoid(sum)
                } else {
                    relu(sum)
                });
            }
            current = next.clone();
            activations.push(current.clone());
        }

        (current[0], activations)
    }

    /// Predict the reward probability for a feature vector.
    pub fn predict(&self, input: &[f64]) -> f64 {
        self.forward(input).0
    }

    /// Online training: one backpropagation pass on a single sample.
    /// target = 1.0 for success, 0.0 for failure.
    pub fn train(&mut self, input: &[f64], target: f64) {
        let (output, activations) = self.forward(input);

        // Output layer: single neuron, sigmoid + MSE loss
        // dE/dz_output = (y - target) * sigmoid'(y)
        let mut delta: Vec<f64> = vec![(output - target) * sigmoid_derivative(output)];

        // Gradient storage (same shapes as layers)
        let mut weight_grads: Vec<Vec<Vec<f64>>> = self
            .layers
            .iter()
            .map(|l| vec![vec![0.0; l.weights[0].len()]; l.weights.len()])
            .collect();
        let mut bias_grads: Vec<Vec<f64>> = self
            .layers
            .iter()
            .map(|l| vec![0.0; l.biases.len()])
            .collect();

        // Backpropagate from output layer to input
        for i in (0..self.layers.len()).rev() {
            let layer_input = &activations[i]; // input to this layer

            // Accumulate gradients for this layer
            for (out_idx, _) in self.layers[i].weights.iter().enumerate() {
                bias_grads[i][out_idx] = delta[out_idx];
                for (in_idx, &a) in layer_input.iter().enumerate() {
                    weight_grads[i][out_idx][in_idx] = delta[out_idx] * a;
                }
            }

            // Propagate delta to the previous layer
            if i > 0 {
                let prev_size = self.layers[i].weights[0].len();
                let mut new_delta = vec![0.0; prev_size];
                for in_idx in 0..prev_size {
                    let mut sum = 0.0;
                    for out_idx in 0..self.layers[i].weights.len() {
                        sum += self.layers[i].weights[out_idx][in_idx] * delta[out_idx];
                    }
                    // activations[i][in_idx] is the post-ReLU value;
                    // relu'(relu(z)) = 1 iff relu(z) > 0, which is correct.
                    new_delta[in_idx] = sum * relu_derivative(activations[i][in_idx]);
                }
                delta = new_delta;
            }
        }

        // Apply gradient descent update
        for (i, layer) in self.layers.iter_mut().enumerate() {
            for (out_idx, weights_row) in layer.weights.iter_mut().enumerate() {
                for (in_idx, w) in weights_row.iter_mut().enumerate() {
                    *w -= self.learning_rate * weight_grads[i][out_idx][in_idx];
                }
                layer.biases[out_idx] -= self.learning_rate * bias_grads[i][out_idx];
            }
        }
    }

    /// Human-readable summary of layer shapes and weight statistics.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let in_size = layer.weights[0].len();
            let out_size = layer.weights.len();
            let total = in_size * out_size;

            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            let mut sum = 0.0;
            for row in &layer.weights {
                for &w in row {
                    if w < min { min = w; }
                    if w > max { max = w; }
                    sum += w;
                }
            }
            let mean = if total > 0 { sum / total as f64 } else { 0.0 };

            let layer_type = if i == self.layers.len() - 1 { "output" } else { "hidden" };
            s.push_str(&format!(
                "  Layer {} ({}): {}x{} = {} weights | min={:.4} max={:.4} mean={:.4}\n",
                i, layer_type, in_size, out_size, total, min, max, mean
            ));
        }
        s.push_str(&format!("  Learning rate: {}\n", self.learning_rate));
        s
    }

    /// Total parameter count across all layers.
    pub fn param_count(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.len() * l.weights[0].len() + l.biases.len())
            .sum()
    }
}