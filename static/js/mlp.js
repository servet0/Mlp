/* ========================================
   mlp.js - MLP Neural Network Engine
   XOR: [2, 2, 1] architecture
   
   Math reference:
     Forward:  net_j = Σ w_ji * y_i + b_j
               y_j   = φ(net_j)
     Error:    e_j(n) = d_j(n) - y_j(n)
     Output δ: δ_j = e_j · φ'(net_j)
     Hidden δ: δ_j = φ'(net_j) · Σ δ_k · w_kj
     Update:   Δw_ji(n) = η·δ_j·y_i + α·Δw_ji(n-1)
   ======================================== */

class MLP {
    constructor() {
        this.layers = [2, 2, 1];
        this.activationFn = 'sigmoid';  // 'sigmoid' or 'tanh'
        this.learningRate = 0.5;
        this.momentum = 0.0;
        this.initWeights();
    }

    initWeights() {
        // Xavier-ish initialization
        this.weights = [];  // weights[l][j][i] = weight from neuron i in layer l to neuron j in layer l+1
        this.biases = [];   // biases[l][j] = bias for neuron j in layer l+1
        this.prevDeltaW = [];
        this.prevDeltaB = [];

        for (let l = 0; l < this.layers.length - 1; l++) {
            const fanIn = this.layers[l];
            const fanOut = this.layers[l + 1];
            const limit = Math.sqrt(6 / (fanIn + fanOut));

            this.weights[l] = [];
            this.biases[l] = [];
            this.prevDeltaW[l] = [];
            this.prevDeltaB[l] = [];

            for (let j = 0; j < fanOut; j++) {
                this.weights[l][j] = [];
                this.prevDeltaW[l][j] = [];
                this.biases[l][j] = (Math.random() * 2 - 1) * limit;
                this.prevDeltaB[l][j] = 0;

                for (let i = 0; i < fanIn; i++) {
                    this.weights[l][j][i] = (Math.random() * 2 - 1) * limit;
                    this.prevDeltaW[l][j][i] = 0;
                }
            }
        }

        // Store forward pass state
        this.nets = [];    // net values before activation
        this.outputs = []; // activations per layer
        this.deltas = [];  // local gradients per layer (excluding input)
        this.errors = [];  // error at output layer
    }

    activation(v) {
        if (this.activationFn === 'tanh') {
            return Math.tanh(v);
        }
        // Sigmoid
        return 1.0 / (1.0 + Math.exp(-v));
    }

    activationDerivative(y) {
        // Derivative expressed in terms of output y
        if (this.activationFn === 'tanh') {
            // φ'(v) = 1 - y²
            return 1.0 - y * y;
        }
        // Sigmoid: φ'(v) = y(1 - y)
        return y * (1.0 - y);
    }

    forward(input) {
        this.nets = [];
        this.outputs = [];
        this.outputs[0] = [...input];

        for (let l = 0; l < this.layers.length - 1; l++) {
            this.nets[l] = [];
            this.outputs[l + 1] = [];

            for (let j = 0; j < this.layers[l + 1]; j++) {
                let net = this.biases[l][j];
                for (let i = 0; i < this.layers[l]; i++) {
                    net += this.weights[l][j][i] * this.outputs[l][i];
                }
                this.nets[l][j] = net;
                this.outputs[l + 1][j] = this.activation(net);
            }
        }

        return this.outputs[this.outputs.length - 1];
    }

    backward(target) {
        const numLayers = this.layers.length;
        this.deltas = [];
        this.errors = [];

        // Output layer deltas
        const outLayer = numLayers - 1;
        const outLayerIdx = outLayer - 1; // index into weights array
        this.deltas[outLayerIdx] = [];
        this.errors = [];

        for (let j = 0; j < this.layers[outLayer]; j++) {
            const y = this.outputs[outLayer][j];
            const e = target[j] - y; // e_j(n) = d_j(n) - y_j(n)
            this.errors[j] = e;
            const phiPrime = this.activationDerivative(y);
            this.deltas[outLayerIdx][j] = e * phiPrime; // δ_j = e_j · φ'(v_j)
        }

        // Hidden layer deltas (back-propagate)
        for (let l = numLayers - 2; l >= 1; l--) {
            const layerIdx = l - 1;
            this.deltas[layerIdx] = [];

            for (let j = 0; j < this.layers[l]; j++) {
                const y = this.outputs[l][j];
                const phiPrime = this.activationDerivative(y);

                let sum = 0;
                for (let k = 0; k < this.layers[l + 1]; k++) {
                    sum += this.deltas[l][k] * this.weights[l][k][j];
                }

                this.deltas[layerIdx][j] = phiPrime * sum; // δ_j = φ'(v_j) · Σ δ_k · w_kj
            }
        }
    }

    updateWeights() {
        const eta = this.learningRate;
        const alpha = this.momentum;

        for (let l = 0; l < this.layers.length - 1; l++) {
            for (let j = 0; j < this.layers[l + 1]; j++) {
                const delta_j = this.deltas[l][j];

                for (let i = 0; i < this.layers[l]; i++) {
                    const y_i = this.outputs[l][i];
                    // Δw_ji(n) = η · δ_j · y_i + α · Δw_ji(n-1)
                    const dw = eta * delta_j * y_i + alpha * this.prevDeltaW[l][j][i];
                    this.weights[l][j][i] += dw;
                    this.prevDeltaW[l][j][i] = dw;
                }

                // Bias update
                const db = eta * delta_j + alpha * this.prevDeltaB[l][j];
                this.biases[l][j] += db;
                this.prevDeltaB[l][j] = db;
            }
        }
    }

    trainSample(input, target) {
        this.forward(input);
        this.backward(target);
        this.updateWeights();

        // Return squared error for this sample: (1/2) * Σ e_j²
        let se = 0;
        for (let j = 0; j < this.errors.length; j++) {
            se += 0.5 * this.errors[j] * this.errors[j];
        }
        return se;
    }

    trainEpoch(trainingData) {
        let totalError = 0;
        for (const sample of trainingData) {
            const input = sample.slice(0, this.layers[0]);
            const target = sample.slice(this.layers[0]);
            totalError += this.trainSample(input, target);
        }
        // E_av = (1/N) * Σ error
        return totalError / trainingData.length;
    }

    // Get all neuron info for visualization
    getNetworkState() {
        const state = {
            layers: this.layers,
            weights: this.weights,
            biases: this.biases,
            outputs: this.outputs,
            nets: this.nets,
            deltas: this.deltas,
            errors: this.errors
        };
        return state;
    }

    // Get delta for a specific neuron (layer, index)
    // layerIndex: 1 = hidden, 2 = output (for [2,2,1])
    getDelta(layerIndex, neuronIndex) {
        if (layerIndex === 0) return null; // Input layer has no delta
        const deltaIdx = layerIndex - 1;
        if (this.deltas[deltaIdx] && this.deltas[deltaIdx][neuronIndex] !== undefined) {
            return this.deltas[deltaIdx][neuronIndex];
        }
        return null;
    }

    getOutput(layerIndex, neuronIndex) {
        if (this.outputs[layerIndex] && this.outputs[layerIndex][neuronIndex] !== undefined) {
            return this.outputs[layerIndex][neuronIndex];
        }
        return null;
    }

    getNet(layerIndex, neuronIndex) {
        if (layerIndex === 0) return null;
        const netIdx = layerIndex - 1;
        if (this.nets[netIdx] && this.nets[netIdx][neuronIndex] !== undefined) {
            return this.nets[netIdx][neuronIndex];
        }
        return null;
    }
}
