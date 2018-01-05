import numpy as np 
from copy import deepcopy 

def train(model, train_dat, test_dat, epochs):

    # train 
    X_train, y_train = train_dat 
    X_test, y_test   = test_dat 
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=200)
    history_dict = deepcopy(history.history) 
    
    # testing 
    scores = model.evaluate(X_test, y_test, verbose=0)
    test_error = 100 - scores[1]*100 
  
    return test_error, history_dict 


