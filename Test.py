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
bindir = 'bin'

def ensure_dir(f):
    if not os.path.exists(bindir):
        os.makedirs(f)

###############################################################################
# Dataset
###############################################################################

os.chdir(wkdir)

# Load the dataset.
#
with open('mnist.pkl', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    mnistdata = u.load()
    
# mnistdata[0]) is train
# mnistdata[1]) is test
# mnistdata[2]) is validate  
#
testfeet = pd.DataFrame(mnistdata[1][0])
testlabl = pd.DataFrame(mnistdata[1][1])

mnistdata = None

# Size of the dataset.
#
N_datapoints = testfeet.shape[0]


##########################################################################################
# Settings
##########################################################################################

# Number of neurons in each layer.
#
N1 = 28 ** 2
N2 = 500
N3 = 500
N4 = 500
N5 = 10

# Load neural network parameters.
#
b1 = readcsv("bin/train_b1.csv")
W12 = readcsv("bin/train_W12.csv")
b2 = readcsv("bin/train_b2.csv")
W23 = readcsv("bin/train_W23.csv")
b3 = readcsv("bin/train_b3.csv")
W34 = readcsv("bin/train_W34.csv")
b4 = readcsv("bin/train_b4.csv")
W45 = readcsv("bin/train_W45.csv")
b5 = readcsv("bin/train_b5.csv")

##########################################################################################
# Methods
##########################################################################################

# Activation functions.
#
sigmoid(x) = 1.0./(1.0+exp(-x))
softplus(x) = log(1.0+exp(x))
softmax(x) = exp(x)./sum(exp(x))

##########################################################################################
# Test
##########################################################################################

# Print header.
#
println("RESPONSES")

# Track percentage of guesses that are correct.
#
N_correct = 0.0
N_tries = 0.0

# Classify each item in the dataset.
#
for i = 1:N_datapoints

    # Load the input.
    #
    x = 6.0*features[i,:]'-3.0

    # Feedforward pass for computing the output.
    #
    y1 = sigmoid(x+b1)
    y2 = softplus(W12'*y1+b2)
    y3 = softplus(W23'*y2+b3)
    y4 = softplus(W34'*y3+b4)
    y5 = softmax(W45'*y4.+b5)

    # Update percentage of guesses that are correct.
    #
    guess = findmax(y5)[2]-1
    answer = round(Int, labels[i])
    if guess == answer
        N_correct += 1.0
    end
    N_tries += 1.0

    # Print response.
    #
    println("  i = $(i), Guess = $(guess), Answer = $(answer)")

end

##########################################################################################
# Results
##########################################################################################

# Print progress report.
#
println("SCORE")
println("  Correct = $(round(100.0*N_correct/N_tries, 5))%")
println("")

