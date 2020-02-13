#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:09:26 2020
@author: UM-AD\huwen
"""

import numpy as np
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
        
    def grad(self, x, delta):
        #calculation of the gradient of the quadratic loss with respect to the weight and bias parameters#
        #the gardients with respect to the biases are the error vectors#
        gradbias=[]
        for i in range(self.L+1):
            gradbias.append(delta[i])        

        #calculate the gradients with respect to the weight vectors#
        #initialize the gradient with respect to the weights, this will be an array of the same size as the weights#
        gradweight=[]
        #the initial layer#
        gradweight.append(np.array([[float(delta[0][j])*float(x) for k in range(1)] for j in range(self.n[0])]))
        for l in range(self.L):
            #layer index is l+1#
            if l==self.L-1:
                #the last layer#
                gradweight.append(np.array([[float(delta[l+1][j])*float(self.outputsequence[l][k]) for k in range(self.n[l])] for j in range(1)]))
            else:
                #all hidden layers except the last layer#
                gradweight.append(np.array([[float(delta[l+1][j])*float(self.outputsequence[l][k]) for k in range(self.n[l])] for j in range(self.n[l+1])]))

        return gradweight, gradbias
        

if __name__ == "__main__":
    #number of hidden layers#
    L=3
    #network size for each hidden layer n[0]=n_1, ..., n[L-1]=n_L#
    n=np.random.randint(1, 5, size=L)
    #activation function#
    sigma=Sigmoid() 
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
    gradweight, gradbias=backprop.grad(x, delta)
    print("weight=", weight)
    print("bias=", bias)    
    print("n=", n)
    print("delta=", delta)
    print("gradweight=", gradweight)
    print("gradbias=", gradbias)

