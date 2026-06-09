import os
import gzip
import struct
import urllib.request
import numpy as np

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}


def _download(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Baixando {os.path.basename(dest_path)}...")
        urllib.request.urlretrieve(url, dest_path)


def _load_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def _load_labels(path):
    with gzip.open(path, "rb") as f:
        struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def load_mnist(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    for key, fname in FILES.items():
        dest = os.path.join(data_dir, fname)
        _download(MNIST_URL + fname, dest)
        paths[key] = dest

    X_train = _load_images(paths["train_images"])
    y_train = _load_labels(paths["train_labels"])
    X_test  = _load_images(paths["test_images"])
    y_test  = _load_labels(paths["test_labels"])

    return (X_train, y_train), (X_test, y_test)
