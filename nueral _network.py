import numpy as np
from tqdm import tqdm 

class NeuralNetwork:
    def __init__(self, layer_sizes, activations, search_hyperparameters=True):
        self._validate_inputs(layer_sizes, activations)
        self.base_layer_sizes = layer_sizes
        self.base_activations = activations
        self.search_hyperparameters = search_hyperparameters
        self.epsilon = 1e-8
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.lr = 0.001
        self.epochs = 100
        self.batch_size = 32
        self.optimizer = 'adam'
        self.early_stopping = False
        self.patience = 5
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.activation_funcs = {
            "relu": (self._relu, self._relu_deriv),
            "elu": (self._elu, self._elu_deriv)
        }
        self._init_weights()

    def _validate_inputs(self, layer_sizes, activations):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output layers.")
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("The number of activations must be one less than the number of layers.")
        if not all(act in ['relu', 'elu'] for act in activations):
            raise ValueError("Supported activations are 'relu' and 'elu'.")

    def _init_weights(self):
        self.params = {}
        self.m = {}
        self.v = {}
        for i in range(1, len(self.base_layer_sizes)):
            fan_in = self.base_layer_sizes[i-1]
            size = (fan_in, self.base_layer_sizes[i])
            self.params[f'W{i}'] = np.random.randn(*size) * np.sqrt(2.0 / fan_in)
            self.params[f'b{i}'] = np.zeros((1, self.base_layer_sizes[i]))
            self.m[f'W{i}'] = np.zeros_like(self.params[f'W{i}'])
            self.v[f'W{i}'] = np.zeros_like(self.params[f'W{i}'])

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_deriv(self, x):
        return (x > 0).astype(float)

    def _elu(self, x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def _elu_deriv(self, x, alpha=1.0):
        return np.where(x > 0, 1, self._elu(x, alpha) + alpha)

    def _softmax(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def _cross_entropy(self, pred, true):
        pred = np.clip(pred, 1e-12, 1 - 1e-12)
        return -np.mean(np.sum(true * np.log(pred), axis=1))

    def _forward(self, X):
        caches = {}
        A = X
        for i in range(1, len(self.base_layer_sizes)):
            Z = A.dot(self.params[f'W{i}']) + self.params[f'b{i}']
            if i < len(self.base_layer_sizes) - 1:
                activation_func = self.activation_funcs[self.base_activations[i-1]][0]
                A = activation_func(Z)
            else:
                A = self._softmax(Z)
            caches[i] = (A, Z)
        return A, caches

    def _backward(self, X, Y, caches):
        grads = {}
        m = X.shape[0]
        A_last, Z_last = caches[len(self.base_layer_sizes)-1]
        dZ = A_last - Y
        for i in reversed(range(1, len(self.base_layer_sizes))):
            A_prev = X if i == 1 else caches[i-1][0]
            grads[f'dW{i}'] = A_prev.T.dot(dZ) / m
            grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / m
            if i > 1:
                W = self.params[f'W{i}']
                dA_prev = dZ.dot(W.T)
                Z_prev = caches[i-1][1]
                activation_deriv = self.activation_funcs[self.base_activations[i-2]][1]
                dZ = dA_prev * activation_deriv(Z_prev)
        return grads

    def _update_params(self, grads, t):
        for i in range(1, len(self.base_layer_sizes)):
            for param in ['W','b']:
                key = f'{param}{i}'
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[f'd{key}']
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[f'd{key}'] ** 2)
                m_corr = self.m[key] / (1 - self.beta1 ** t)
                v_corr = self.v[key] / (1 - self.beta2 ** t)
                self.params[key] -= self.lr * m_corr / (np.sqrt(v_corr) + self.epsilon)

    def _one_hot(self, y, num_classes):
        y_enc = np.zeros((y.size, num_classes))
        for idx, val in enumerate(y):
            y_enc[idx, val-1] = 1
        return y_enc

    def _adjust_learning_rate(self, epoch):
        """Adjust learning rate dynamically."""
        if epoch % 10 == 0:
            self.lr *= 0.9
            print(f"Learning rate adjusted to {self.lr:.6f}")

    def train(self, X, y, validation_data=None):
        n_samples = X.shape[0]
        num_classes = self.base_layer_sizes[-1]
        self._init_weights()
        
        for epoch in tqdm(range(1, self.epochs + 1), desc="Training Progress"):
            idx = np.random.permutation(n_samples)
            X_shuffled = X[idx]
            y_shuffled = y[idx]
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                y_enc = self._one_hot(y_batch, num_classes)
                A, caches = self._forward(X_batch)
                loss = self._cross_entropy(A, y_enc)
                grads = self._backward(X_batch, y_enc, caches)
                self._update_params(grads, epoch)

            if validation_data:
                val_loss = self.evaluate(*validation_data)
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            self._adjust_learning_rate(epoch)
            if epoch % max(1, self.epochs // 10) == 0:
                print(f'Epoch {epoch} loss {loss:.4f}')

    def evaluate(self, X, y):
        y_enc = self._one_hot(y, self.base_layer_sizes[-1])
        A, _ = self._forward(X)
        return self._cross_entropy(A, y_enc)

    def predict(self, X):
        A, _ = self._forward(X)
        return np.argmax(A, axis=1) + 1