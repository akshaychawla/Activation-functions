"""
CNN with sin activation 
author: Akshay Chawla
email: chawla.akshay1234@gmail.com
"""

import numpy
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import Dropout
from keras.layers import Flatten, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K

def create_model():

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
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs=model_ip, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print model.summary()
    return model
