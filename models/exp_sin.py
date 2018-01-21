"""
CNN with exp(-abs(x)) * sin(x) activation 
author: Akshay Chawla
email: chawla.akshay1234@gmail.com
"""

import numpy
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Dropout
from keras.layers import Flatten, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K

def _exp_sin_fun(ip):
    return K.exp(-K.abs(ip)) * K.sin(ip) 

def create_model():

    model_ip = Input(shape=(1,28,28))
    x = Conv2D(30, (5, 5))(model_ip)
    x = Lambda(_exp_sin_fun)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(15, (3, 3))(x)
    x = Lambda(_exp_sin_fun)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = Lambda(_exp_sin_fun)(x)
    x = Dense(50)(x)
    x = Lambda(_exp_sin_fun)(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs=model_ip, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print model.summary()
    return model
