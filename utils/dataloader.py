import numpy as np 
from keras.datasets import mnist 
from keras.utils import np_utils 

def get_mnist():

    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)

    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    return (X_train, y_train), (X_test, y_test), num_classes
