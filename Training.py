# -*- coding: utf-8 -*-
"""
Not Author: scuba-man-spiff
Date: 12/31/2015
Language: Python 3.4.3

This is a learning exercise creating a Python conversion of Jared L. Ostmeyer's 
Julia DNN classifier.  This is for learning purposes only to help myself become
more familiar both with Neural nets and with the Julia language
       Original found at:  https://github.com/jostmey/DeepNeuralClassifier
       
Data set found at: http://deeplearning.net/tutorial/gettingstarted.html
"""

###############################################################################
# Packages
###############################################################################

import numpy as np
import pandas as pd
import scipy as sp
import os
import sys
import re
import pickle
import logging
import math
import random

###############################################################################
# Blobal Vars (Like Global Vars, but fatter)
###############################################################################

wkdir = 'D:/UserFiles/GitHub/Python3---Scratch-Deep-Neural-Net-Classifier'

###############################################################################
# Dataset
###############################################################################

os.chdir(wkdir)

with open('mnist.pkl', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    mnistdata = u.load()
    
# mnistdata[0]) is train
# mnistdata[1]) is test
# mnistdata[2]) is validate
    
trainfeet = pd.DataFrame(mnistdata[0][0])
trainlabl = pd.DataFrame(mnistdata[0][1])

mnistdata = None
###############################################################################
# Check max and min
# np.amax(trainfeet, axis=0)
# np.amax(trainfeet, axis=0)
###############################################################################

####
# Commented out from Julia code - might need to use
# # Scale feature values to be between 0 and 1.
# #
# features = data[1]
# features /= 255.0
####

# Size of the dataset.
#
N_datapoints = trainfeet.shape[0]

###############################################################################
# Settings
###############################################################################

# Schedule for updating the neural network.
#
N_minibatch = 100
N_updates = round(N_datapoints/N_minibatch)*500

# Number of neurons in each layer.
#
N1 = 28**2  ## Because 28 x 28 pixels in training images
N2 = 500
N3 = 500
N4 = 500
N5 = 10

# Initialize random (normal) neural network parameters.
#
b1 = 0.1 * np.random.normal(size = (N1))
W12 = 0.1 * np.random.normal(size = (N1, N2))
b2 = 0.1 * np.random.normal(size = (N2))
W23 = 0.1 * np.random.normal(size = (N2, N3))
b3 = 0.1 * np.random.normal(size = (N3))
W34 = 0.1 * np.random.normal(size = (N3, N4))
b4 = 0.1 * np.random.normal(size = (N4))
W45 = 0.1 * np.random.normal(size = (N4, N5))
b5 = 0.1 * np.random.normal(size = (N5))

# Initial learning rate.
#
alpha = 0.001

# Dropout probability for removing neurons.
#
dropout = 0.5

# Momentum factor.
#
momentum = 0.75

###############################################################################
# Methods
###############################################################################

# Generate mask for neuron dropout.
#
def removeable(n):
    return 1.0 * (dropout <= np.random.rand(n))
    
# Activation functions.
#
def sigmoid(n):
    return [1.0 / (1.0 + math.exp(-x)) for x in n]

def softplus(n):
    return [math.log(1.0 + math.exp(x)) for x in n]

def softmax(n):
    t = sum(math.exp(x))
    return [math.exp(x) / t for x in n]

# Derivative of each activation function given the output.
#
def dsigmoid(m):
    return [y * (1.0 - y) for y in m]
    
def dsoftplus(m):
    return [(1.0 - math.exp(-y)) for y in m]
    
def dsoftmax(m):
    return [y * (1.0 - y) for y in m]

###############################################################################
# Train
###############################################################################

# Holds change in parameters from a minibatch.
#
db1  = np.zeros(N1)
dW12 = np.zeros((N1, N2))
db2  = np.zeros(N2)
dW23 = np.zeros((N2, N3))
db3  = np.zeros(N3)
dW34 = np.zeros((N3, N4))
db4  = np.zeros(N4)
dW45 = np.zeros((N4, N5))
db5  = np.zeros(N5)

# Track percentage of guesses that are correct.
#
N_correct = 0.0
N_tries = 0.0

# Repeatedly update parameters.
#
for i in range(1, N_updates):

    # test
    # i = 1

    # Generate masks for thinning out neural network (dropout procedure).
    #
    r2 = removeable(N2)
    r3 = removeable(N3)
    r4 = removeable(N4)

    # Collect multiple updates for minibatch.
    #
    for j in range(1, N_minibatch):

        # test
        # j = 1

        # Randomly load item from the dataset (part of stochastic gradient descent).
        #
        k = rand(1:N_datapoints)

        x = 6.0*features[k,:]'-3.0

        z = zeros(10)
        z[round(Int, labels[k])+1] = 1.0

        # Feedforward pass for computing the output.
        #
        y1 = sigmoid(x+b1)
        y2 = softplus(W12'*y1+b2).*r2
        y3 = softplus(W23'*y2+b3).*r3
        y4 = softplus(W34'*y3+b4).*r4
        y5 = softmax(W45'*y4+b5)

        # Backpropagation for computing the gradients of the Likelihood function.
        #
        e5 = z-y5
        e4 = (W45*e5).*dsoftplus(y4).*r4
        e3 = (W34*e4).*dsoftplus(y3).*r3
        e2 = (W23*e3).*dsoftplus(y2).*r2
        e1 = (W12*e2).*dsigmoid(y1)

        # Add the errors to the minibatch.
        #
        db1  = db1  + e1
        dW12 = dW12 + y1*e2'
        db2  = db2  + e2
        dW23 = DW23 + y2*e3'
        db3  = db3  + e3
        dW34 = dW34 + y3*e4'
        db4  = db4  + e4
        dW45 = DW45 + y4*e5'
        db5  = db4  + e5

        # Update percentage of guesses that are correct.
        #
        guess = findmax(y5)[2]-1
        answer = findmax(z)[2]-1
        if guess == answer
            N_correct += 1.0
        end
        N_tries += 1.0

    end

    # Update parameters using stochastic gradient descent.
    #
    b1  = b1  + alpha * db1
    W12 = W12 + alpha * dW12
    b2  = b2  + alpha * db2
    W23 = W23 + alpha * dW23
    b3  = b3  + alpha * db3
    W34 = W34 + alpha * dW34
    b4  = b4  + alpha * db4
    W45 = W45 + alpha * dW45
    b5  = b5  + alpha * db5

    # Reset the parameter changes from the minibatch (scale by momentum factor).
    #
    db1  = db1  * momentum
    dW12 = bW12 * momentum
    db2  = db2  * momentum
    dW23 = bW23 * momentum
    db3  = db3  * momentum
    dW34 = dW34 * momentum
    db4  = db4  * momentum
    dW45 = dW45 * momentum
    db5  = db5  * momentum

    # Decrease the learning rate (part of stochastic gradient descent).
    #
    alpha = alpha * (N_updates-i)/(N_updates-i+1)

    # Periodic checks.
    #
    if i % 100 == 0:

        # Print progress report.
        #
        println("REPORT")
        println("  Batch = $(round(Int, i))")
        println("  alpha = $(round(alpha, 8))")
        println("  Correct = $(round(100.0*N_correct/N_tries, 8))%")
        println("")
        flush(STDOUT)

    # Reset percentage of guesses that are correct.
    #
    N_tries = 0.0
    N_correct = 0.0


# Scale effected weights by probability of undergoing dropout.
#
W23 = W23 * (1.0 - dropout)
W34 = W34 * (1.0 - dropout)
W45 = W45 * (1.0 - dropout)

###############################################################################
# Save
###############################################################################

# Create folder to hold parameters.
#
mkpath("bin")

# Save the parameters.
#
writecsv("bin/train_b1.csv", b1)
writecsv("bin/train_W12.csv", W12)
writecsv("bin/train_b2.csv", b2)
writecsv("bin/train_W23.csv", W23)
writecsv("bin/train_b3.csv", b3)
writecsv("bin/train_W34.csv", W34)
writecsv("bin/train_b4.csv", b4)
writecsv("bin/train_W45.csv", W45)
writecsv("bin/train_b5.csv", b5)
