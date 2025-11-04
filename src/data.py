import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, fetch_openml

def load_and_preprocess_iris(test_size=0.2, random_state=42):
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def load_and_preprocess_mnist(test_size=0.2, random_state=42):
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def load_and_preprocess_fashion_mnist(test_size=0.2, random_state=42):
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1)
    X = fashion_mnist.data.astype(np.float32) / 255.0
    y = fashion_mnist.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def load_and_preprocess_cifar10(test_size=0.2, random_state=42):
    cifar10 = fetch_openml('CIFAR_10_small', version=1)
    X = cifar10.data.astype(np.float32) / 255.0
    y = cifar10.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # Reshape to image dimensions: (samples, height, width, channels)
    X_train = X_train.reshape(-1, 32, 32, 3)
    X_test = X_test.reshape(-1, 32, 32, 3)
    return X_train, X_test, y_train, y_test