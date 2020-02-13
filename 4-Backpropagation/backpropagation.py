#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:09:26 2020
@author: UM-AD\huwen
"""

import numpy as np
import matplotlib.pyplot as plt

from activations import Sigmoid, ReLU, Tanh, Exponential
from fullnetwork import onelayer, fullnetwork

class backpropagation(object):
    
    def __init__(self,
                 L=1, #number of hidden layers#
                 n=np.random.randint(1, 6, size=1), #network size for each hidden layer n[0]=n_1, ..., m[L-1]=n_L#
                 activation=Sigmoid(),
                 weight=[],
                 bias=[]
                 ):
        self.L=L
        self.n=n
        self.activation=activation
        self.weight=weight
        self.bias=bias
    
    def error(self, x, y):
        #calculating the error function via backpropagation#
        delta=[]
        for i in range(self.L):
            delta.append(np.zeros(shape=[self.n[i],1]))
        return delta
        
        
    def grad(self, x, y):
        #calculation of the gradient of the quadratic loss with respect to the weight and bias parameters#
        return 0
        

if __name__ == "__main__":
    #number of hidden layers#
    L=3
    #network size for each hidden layer n[0]=n_1, ..., n[L-1]=n_L#
    n=np.random.randint(1, 10, size=L)
    #activation function#
    sigma=Tanh() 
    #set the network#
    network=fullnetwork(L=L, n=n, activation=sigma)
    #set the initial weight and bias#
    weight, bias=network.setparameter()
    #set the backpropagation calculations#
    backprop=backpropagation(L=L, n=n, activation=sigma, weight=weight, bias=bias)
    delta=backprop.error(1,1)
    print("n=", n)
    print("delta=", delta)

