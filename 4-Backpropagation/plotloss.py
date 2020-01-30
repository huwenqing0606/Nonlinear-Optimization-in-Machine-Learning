# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:54:10 2020

@author: Wenqing Hu (Missouri S&T)
"""

import numpy as np
import matplotlib.pyplot as plt
import random

from activations import Sigmoid, ReLU, Tanh, Exponential
from fullnetwork import onelayer, fullnetwork
from mpl_toolkits.mplot3d import Axes3D

outputfile = open('fullnetworkoutput_ReLU.txt', 'w') 


#number of hidden layers#
L=3
#network size for each hidden layer n[0]=n_1, ..., n[L-1]=n_L#
n=np.random.randint(1, 10, size=L)
#training set size#     
training_size=1 
#(N, N) meshgrid#
N=100
#activation function#
sigma=ReLU() 

#set the network#
network=fullnetwork(L=L, n=n, activation=sigma)
#set the initial weight and bias#
weight, bias=network.setparameter()


#choose one layer from from [1, L-1]#
weightindex_startlayer=np.random.randint(1, L, size=None)
#its next layer#
weightindex_nextlayer=weightindex_startlayer+1
#the two weights taken from randomly sample two neurons from each of the above layers, from [1, width of that layer]#
weightindex_neuron_startlayer=np.random.randint(1, n[weightindex_startlayer-1]+1, size=2)
weightindex_neuron_nextlayer=np.random.randint(1, n[weightindex_nextlayer-1]+1, size=2)


#set training data#
X=[]
for m in range(training_size):
    X.append(np.random.normal(0,1,1))
    
Y=[]
for m in range(training_size):
    Y.append(np.random.normal(0,1,1))

#plot the loss#
def plot_network_loss():
    a_1 = np.linspace(-10, 10, N)
    a_2 = np.linspace(-10, 10, N)
    L = [[0 for i in range(N)] for j in range(N)]
    for i in range(N):
        for j in range (N):
            #set the particular two weights to be a_1[i] and a_2[j]#
            weight[weightindex_startlayer][weightindex_neuron_nextlayer[0]-1][weightindex_neuron_startlayer[0]-1]=a_1[i]
            weight[weightindex_startlayer][weightindex_neuron_nextlayer[1]-1][weightindex_neuron_startlayer[1]-1]=a_2[j]
            #calculate the mean square error produced by the weight parameters at (a_1[i], a_2[j])#
            Z=[]
            for m in range(training_size):
                Z.append((Y[m]-float(network.output(float(X[m]), weight, bias)))**2)                
            L[i][j]=0.5*np.mean(np.array(Z))
                              
    fig = plt.figure()
    ax = Axes3D(fig)
    u=np.array(a_1)
    v=np.array(a_2)
    w=np.array(L)
    u, v = np.meshgrid(u, v)
    ax.plot_surface(u, v, w, rstride=1, cstride=1, cmap='rainbow')
    plt.show()


if __name__ == "__main__":
    plot_network_loss()
    outputfile.close() 