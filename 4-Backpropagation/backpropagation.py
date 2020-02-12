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
                 activation=Sigmoid()
                 ):
        self.L=L
        self.n=n
        self.activation=activation
    
    def error(self, x, y, weight, bias):
        #calculating the error function#
        
        