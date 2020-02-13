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
                 bias=[],
                 outputsequence=[],
                 preoutputsequence=[]
                 ):
        self.L=L
        self.n=n
        self.activation=activation
        self.weight=weight
        self.bias=bias
        self.outputsequence=outputsequence
        self.preoutputsequence=preoutputsequence
    
    def error(self, y):
        #calculating the error function via backpropagation#
        delta=[]
        #initialize delta as zero column vectors with prescribed sizes#
        for i in range(self.L):
            delta.append(np.zeros(shape=[self.n[i],1]))
        delta.append(np.zeros(shape=[1,1]))
        #the last (output) layer delta#
        delta[self.L+1-1]=np.dot(self.activation.grad(self.preoutputsequence[self.L+1-1]), self.outputsequence[self.L+1-1]-np.array(y))
        #backpropagation: from the last layer to the first hidden layer calculate all the error functions#
        for i in reversed(range(self.L)):
            vector1=np.dot(self.weight[i+1].T, delta[i+1])
            vector2=[]
            for index in range(self.n[i]):
                vector2.append(self.activation.grad(self.preoutputsequence[i][index][0]))
            vector2=np.array(vector2).reshape(-1,1)
            delta[i]=vector1*vector2
        return delta
        
    def grad(self, delta):
        #calculation of the gradient of the quadratic loss with respect to the weight and bias parameters#
        gradweight=self.weight
        gradbias=self.bias
        #the gardients with respect to the biases are the error vectors#
        for i in range(self.L+1):
            gradbias[i]=delta[i]
        #calculate the gradients with respect to the weight vectors#
        for i in range(self.L+1):
            if i==0: 
                #initial layer#
                for j in range(self.n[i]):
                    for k in range(1):
                        gradweight[i][j][k]=delta[i][j]*self.outputsequence[i-1][k]
            elif i==self.L:
                #last layer#
                for j in range(1):
                    for k in range(self.n[i-1]):
                        gradweight[i][j][k]=delta[i][j]*self.outputsequence[i-1][k]
            else:
                #all hidden layers#
                for j in range(self.n[i]):
                    for k in range(self.n[i-1]):
                        gradweight[i][j][k]=delta[i][j]*self.outputsequence[i-1][k]
        return gradweight, gradbias
        

if __name__ == "__main__":
    #number of hidden layers#
    L=3
    #network size for each hidden layer n[0]=n_1, ..., n[L-1]=n_L#
    n=np.random.randint(1, 3, size=L)
    #activation function#
    sigma=ReLU() 
    #set the network#
    network=fullnetwork(L=L, n=n, activation=sigma)
    #set the initial weight and bias#
    weight, bias=network.setparameter()
    #calculate the network output and the layer-by-layer output sequence and preoutput sequence#
    x=1
    y=1
    networkoutput, outputsequence, preoutputsequence=network.output(x, weight, bias)
    #set the backpropagation calculations#
    backprop=backpropagation(L=L, 
                             n=n, 
                             activation=sigma, 
                             weight=weight, 
                             bias=bias, 
                             outputsequence=outputsequence, 
                             preoutputsequence=preoutputsequence)
    delta=backprop.error(y)
    gradweight, gradbias=backprop.grad(delta)
    print("n=", n)
    print("delta=", delta)
    print("gradweight=", gradweight)
    print("gradbias=", gradbias)

