"""
Script to test different activation functions on a 
SOTA CNN architecture for mnist dataset. 

author: Akshay Chawla
email: chawla.akshay1234@gmail.com
"""

import numpy
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import Dropout
from keras.layers import Flatten, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# Data loading
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

# import ipdb; ipdb.set_trace()

# define the larger model
def larger_model():

    model_ip = Input(shape=(1,28,28))
    x = Conv2D(30, (5, 5))(model_ip)
    x = Lambda(lambda y: K.sin(y))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(15, (3, 3))(x)
    x = Lambda(lambda y: K.sin(y))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = Lambda(lambda y: K.sin(y))(x)
    x = Dense(50)(x)
    x = Lambda(lambda y: K.sin(y))(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=model_ip, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print model.summary()
    return model

model = larger_model()
# Fit the model 
history = model.fit(X_train, y_train, validation_split=0.1, epochs=2, batch_size=200)

with open("./results/sin_x.pkl", "wb") as f: 
    import pickle 
    pickle.dump(history.history, f) 

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Error: %.2f%%" % (100-scores[1]*100))
