import numpy as np
from .activations import ACTIVATIONS, softmax
from .losses import cross_entropy, cross_entropy_softmax_gradient
from .optimizers import SGD


class MLP:
    """
    Multi-Layer Perceptron with arbitrary hidden layers.

    layer_sizes: list of ints, e.g. [784, 256, 128, 10]
                 first = input dim, last = num classes
    activations: list of str for hidden layers, e.g. ["relu", "relu"]
                 length must be len(layer_sizes) - 2
    """

    def __init__(self, layer_sizes, hidden_activation="relu", optimizer=None, seed=42):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1  # number of weight matrices

        act_fn, act_deriv = ACTIVATIONS[hidden_activation]
        self.act_fn = act_fn
        self.act_deriv = act_deriv

        self.params = {}
        self._init_weights()

        self.optimizer = optimizer if optimizer is not None else SGD(learning_rate=0.01)

        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def _init_weights(self):
        """He initialization for ReLU layers."""
        for i in range(self.n_layers):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            # He init: scale by sqrt(2/fan_in) for ReLU; works fine for sigmoid too
            scale = np.sqrt(2.0 / fan_in)
            self.params[f"W{i+1}"] = np.random.randn(fan_in, fan_out) * scale
            self.params[f"b{i+1}"] = np.zeros((1, fan_out))

    def forward(self, X):
        """
        Returns cache needed for backprop.
        cache: list of (A_prev, Z) for each layer
        """
        cache = []
        A = X
        for i in range(self.n_layers - 1):
            Z = A @ self.params[f"W{i+1}"] + self.params[f"b{i+1}"]
            cache.append((A, Z))
            A = self.act_fn(Z)

        # Output layer: linear -> softmax
        i = self.n_layers - 1
        Z_out = A @ self.params[f"W{i+1}"] + self.params[f"b{i+1}"]
        cache.append((A, Z_out))
        A_out = softmax(Z_out)
        return A_out, cache

    def backward(self, y_pred, y_true_onehot, cache):
        """
        Computes gradients for all parameters via backpropagation.
        """
        grads = {}
        n_layers = self.n_layers

        # Gradient at output layer (combined softmax + cross-entropy)
        dZ = cross_entropy_softmax_gradient(y_pred, y_true_onehot)

        for i in reversed(range(n_layers)):
            A_prev, Z = cache[i]

            grads[f"dW{i+1}"] = A_prev.T @ dZ
            grads[f"db{i+1}"] = np.sum(dZ, axis=0, keepdims=True)

            if i > 0:
                # Propagate to previous layer
                dA_prev = dZ @ self.params[f"W{i+1}"].T
                _, Z_prev = cache[i - 1]
                dZ = dA_prev * self.act_deriv(Z_prev)

        return grads

    def _one_hot(self, y, n_classes):
        oh = np.zeros((len(y), n_classes))
        oh[np.arange(len(y)), y] = 1.0
        return oh

    def predict_proba(self, X):
        probs, _ = self.forward(X)
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=20, batch_size=64, verbose=True):
        n_classes = self.layer_sizes[-1]
        n_samples = X_train.shape[0]

        for epoch in range(1, epochs + 1):
            # Shuffle training data each epoch
            perm = np.random.permutation(n_samples)
            X_shuf = X_train[perm]
            y_shuf = y_train[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                X_batch = X_shuf[start:start + batch_size]
                y_batch = y_shuf[start:start + batch_size]
                y_oh = self._one_hot(y_batch, n_classes)

                y_pred, cache = self.forward(X_batch)
                loss = cross_entropy(y_pred, y_oh)
                grads = self.backward(y_pred, y_oh, cache)

                self.params = self.optimizer.update(self.params, grads)

                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            train_acc = self.accuracy(X_train, y_train)
            self.history["train_loss"].append(avg_loss)
            self.history["train_acc"].append(train_acc)

            val_info = ""
            if X_val is not None:
                y_val_oh = self._one_hot(y_val, n_classes)
                val_pred, _ = self.forward(X_val)
                val_loss = cross_entropy(val_pred, y_val_oh)
                val_acc = self.accuracy(X_val, y_val)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                val_info = f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"

            if verbose:
                print(
                    f"Epoch {epoch:3d}/{epochs}  "
                    f"loss={avg_loss:.4f}  train_acc={train_acc:.4f}"
                    + val_info
                )

        return self.history

    def gradient_check(self, X, y, eps=1e-5, sample_size=5):
        """
        Numerical gradient check on a small batch.
        Returns max relative difference between analytical and numerical grads.
        """
        n_classes = self.layer_sizes[-1]
        X_small = X[:sample_size]
        y_small = y[:sample_size]
        y_oh = self._one_hot(y_small, n_classes)

        y_pred, cache = self.forward(X_small)
        grads_analytical = self.backward(y_pred, y_oh, cache)

        max_diff = 0.0
        for key in self.params:
            param = self.params[key]
            grad_key = "d" + key
            grad_analytical = grads_analytical[grad_key]

            # Only check a few elements to keep runtime low
            it = np.nditer(param, flags=["multi_index"])
            checked = 0
            while not it.finished and checked < 20:
                idx = it.multi_index
                orig = param[idx]

                param[idx] = orig + eps
                y_plus, _ = self.forward(X_small)
                loss_plus = cross_entropy(y_plus, y_oh)

                param[idx] = orig - eps
                y_minus, _ = self.forward(X_small)
                loss_minus = cross_entropy(y_minus, y_oh)

                param[idx] = orig  # restore

                grad_num = (loss_plus - loss_minus) / (2 * eps)
                grad_ana = grad_analytical[idx]

                denom = max(abs(grad_num) + abs(grad_ana), 1e-8)
                diff = abs(grad_num - grad_ana) / denom
                max_diff = max(max_diff, diff)

                it.iternext()
                checked += 1

        return max_diff
