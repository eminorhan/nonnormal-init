'''Generate data for RNN experiments'''

import numpy as np
from keras.datasets import mnist

def addition_task(n_train, n_test, T):
    "Generate data from the addition task. Arguments: T (sequence length), n_train, n_test (data sizes)"
    # Train data
    signal_train = np.random.rand(n_train, T, 1)
    mask_train = np.zeros((n_train, T, 1))
    ind1_train = np.random.randint(low=0, high=T//2, size=n_train)
    ind2_train = np.random.randint(low=T//2, high=T, size=n_train)
    mask_train[range(0, n_train), ind1_train] = 1.
    mask_train[range(0, n_train), ind2_train] = 1.

    X_train = np.concatenate((mask_train, signal_train), axis=2)
    Y_train = signal_train[range(0, n_train), ind1_train, :] + signal_train[range(0, n_train), ind2_train, :]

    # Test data
    signal_test = np.random.rand(n_test, T, 1)
    mask_test = np.zeros((n_test, T, 1))
    ind1_test = np.random.randint(low=0, high=T//2, size=n_test)
    ind2_test = np.random.randint(low=T//2, high=T, size=n_test)
    mask_test[range(0, n_test), ind1_test] = 1.
    mask_test[range(0, n_test), ind2_test] = 1.

    X_test = np.concatenate((mask_test, signal_test), axis=2)
    Y_test = signal_test[range(0, n_test), ind1_test, :] + signal_test[range(0, n_test), ind2_test, :]

    return (X_train, Y_train), (X_test, Y_test)

def copy_task(n_train, n_test, T):
    "Generate data from the copy task. Arguments: T (sequence length), n_train, n_test (data sizes)"
    # Train data
    X_train = np.zeros((n_train, T), dtype='int64')
    data_train = np.random.randint(low=1, high=9, size=(n_train, 10))
    X_train[:, :10] = data_train
    X_train[:, -11] = 9
    Y_train = np.zeros((n_train, T), dtype='int64')
    Y_train[:, -10:] = X_train[:, :10]

    # Test data
    X_test = np.zeros((n_test, T), dtype='int64')
    data_test = np.random.randint(low=1, high=9, size=(n_test, 10))
    X_test[:, :10] = data_test
    X_test[:, -11] = 9
    Y_test = np.zeros((n_test, T), dtype='int64')
    Y_test[:, -10:] = X_test[:, :10]

    X_train = (np.arange(X_train.max() + 1) == X_train[..., None]).astype(int)
    X_test = (np.arange(X_test.max() + 1) == X_test[..., None]).astype(int)
    Y_train = (np.arange(Y_train.max() + 1) == Y_train[..., None]).astype(int)
    Y_test = (np.arange(Y_test.max() + 1) == Y_test[..., None]).astype(int)

    return (X_train, Y_train), (X_test, Y_test)

def mnist_task():
    "Generate data for the pixel-by-pixel MNIST task."
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], -1, 1)
    x_test = x_test.reshape(x_test.shape[0], -1, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def permuted_mnist_task():
    "Generate data from the permuted pixel-by-pixel MNIST task."
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], -1, 1)
    x_test = x_test.reshape(x_test.shape[0], -1, 1)

    permuted_ind = np.random.permutation(x_train.shape[1])
    x_train = x_train[:, permuted_ind, :]
    x_test = x_test[:, permuted_ind, :]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)
