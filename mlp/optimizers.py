import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads):
        """
        params: dict {"W1": ..., "b1": ..., ...}
        grads:  dict {"dW1": ..., "db1": ..., ...}
        """
        for key in params:
            grad_key = "d" + key
            if grad_key not in grads:
                continue

            if self.momentum > 0:
                if key not in self.velocity:
                    self.velocity[key] = np.zeros_like(params[key])
                self.velocity[key] = (
                    self.momentum * self.velocity[key] - self.lr * grads[grad_key]
                )
                params[key] += self.velocity[key]
            else:
                params[key] -= self.lr * grads[grad_key]

        return params


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # first moment
        self.v = {}  # second moment
        self.t = 0   # time step

    def update(self, params, grads):
        self.t += 1
        for key in params:
            grad_key = "d" + key
            if grad_key not in grads:
                continue

            g = grads[grad_key]

            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return params
