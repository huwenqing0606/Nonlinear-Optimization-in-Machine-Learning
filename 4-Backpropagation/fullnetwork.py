#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 08:23:40 2020

@author: Wenqing Hu (Missouri S&T)
"""

"""
Construct the output function of a fully connnected neural network 
with L layers and network size n_1, ..., n_L
parameters L, n_1, ..., n_L are given
"""

import numpy as np

L=5 #number of layers#
n=np.random.randint(1, 6, size=L) #network size for each layer n[0]=n_1, ..., m[L-1]=n_L#

"""
one layer of the neural network, input vector x^{in}, output \sigma(W x^{in} + b)
W is weight vector, b is the bias
"""
class onelayer(object):
    
    def __init__(self,
                 inputsize=1,       #input layer size#
                 outputsize=1,      #output layer size#
                 inputvector=[],    #input vector#
                 activation,        #activation function#
                 weight=[],         #weights from input to output layer#
                 bias=[],           #bias vectors in the particular layer#
                 ):
        self.size=size
        




