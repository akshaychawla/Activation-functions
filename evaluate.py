import numpy as np
import matplotlib.pyplot as plt 
import sys, os 
import pickle

# Load all history pickle files 
files = sys.argv[1:]
history = {} 
for filename in files: 
    with open(filename, "rb") as f: 
        keyname = filename.replace("results/","").replace(".pkl","")
        history[keyname] = pickle.load(f) 

# Plots 

# Training Accuracy 
fig, ax = plt.subplots()
for k,v in history.items():
    t_acc = v["acc"]
    ax.plot(range(len(t_acc)), t_acc, label=k)
    ax.legend()
plt.title("Training Accuracy")
plt.show()

# Training Loss 
fig, ax = plt.subplots()
for k,v in history.items():
    t_loss = v["loss"]
    ax.plot(range(len(t_loss)), t_loss, label=k)
    ax.legend()
plt.title("Training Loss")
plt.show()

# Validation Accuracy 
fig, ax = plt.subplots()
for k,v in history.items():
    v_acc = v["val_acc"]
    ax.plot(range(len(v_acc)), v_acc, label=k)
    ax.legend()
plt.title("Validation Accuracy")
plt.show()

# Validation Loss
fig, ax = plt.subplots()
for k,v in history.items():
    v_loss = v["val_loss"]
    ax.plot(range(len(v_loss)), v_loss, label=k)
    ax.legend()
plt.title("Validation Loss")
plt.show()

# import ipdb; ipdb.set_trace() 
