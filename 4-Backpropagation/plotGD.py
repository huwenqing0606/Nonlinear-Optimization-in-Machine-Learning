# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:30:29 2020

@author: huwenqing
"""

import numpy as np
import matplotlib.pyplot as plt

from activations import Sigmoid, ReLU, Tanh, Exponential
from fullnetwork import onelayer, fullnetwork
from backpropagation import backpropagation
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import animation

#number of hidden layers#
L=3
#network size for each hidden layer n[0]=n_1, ..., n[L-1]=n_L#
n=np.random.randint(1, 3, size=L)
#activation function#
sigma=Tanh() 

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


#set training data (x,y)#
x=np.random.normal(0,1,1)
y=np.random.normal(0,1,1)


#plot the gd trajectory via backpropagation#
def plot_gd_trajectory(w1_init, w2_init, learningrate):
    Loss=[]
    w_1=[]
    w_2=[]
    w_1.append(w1_init)
    w_2.append(w2_init)
    weight[weightindex_startlayer][weightindex_neuron_nextlayer[0]-1][weightindex_neuron_startlayer[0]-1]=w1_init
    weight[weightindex_startlayer][weightindex_neuron_nextlayer[1]-1][weightindex_neuron_startlayer[1]-1]=w2_init
    networkoutput, outputsequence, preoutputsequence=network.output(float(x), weight, bias)
    Loss.append(float(0.5*(y-float(networkoutput))**2))
    for i in range(100):
        #calculate the gradient with respect to current weight and bias#
        backprop=backpropagation(L=L, 
                                 n=n, 
                                 activation=sigma, 
                                 weight=weight, 
                                 bias=bias, 
                                 outputsequence=outputsequence, 
                                 preoutputsequence=preoutputsequence)
        delta=backprop.error(y)
        gradweight, gradbias=backprop.grad(delta)
        #update the weights and the loss values#
        weight[weightindex_startlayer][weightindex_neuron_nextlayer[0]-1][weightindex_neuron_startlayer[0]-1]=w_1[i]-learningrate*gradweight[weightindex_startlayer][weightindex_neuron_nextlayer[0]-1][weightindex_neuron_startlayer[0]-1]
        weight[weightindex_startlayer][weightindex_neuron_nextlayer[1]-1][weightindex_neuron_startlayer[1]-1]=w_2[i]-learningrate*gradweight[weightindex_startlayer][weightindex_neuron_nextlayer[1]-1][weightindex_neuron_startlayer[1]-1]
        networkoutput, outputsequence, preoutputsequence=network.output(float(x), weight, bias)
        Loss.append(float(0.5*(y-float(networkoutput))**2))
        w_1.append(weight[weightindex_startlayer][weightindex_neuron_nextlayer[0]-1][weightindex_neuron_startlayer[0]-1])
        w_2.append(weight[weightindex_startlayer][weightindex_neuron_nextlayer[1]-1][weightindex_neuron_startlayer[1]-1])
    
    return w_1, w_2, Loss                          


if __name__ == "__main__":
    w1_init=1 
    w2_init=1
    learningrate=0.01
    w_1, w_2, Loss=plot_gd_trajectory(w1_init, w2_init, learningrate)
    print("w1=", w_1)
    print("w2=", w_2)
    print("Loss=", Loss)


    fig = plt.figure()
    ax=Axes3D(fig)
    line=ax.plot([],[],'b:')
    point=ax.plot([],[],'bo',markersize=10)
    images=[]
    def init():
        line=ax.plot([],[],'b:',markersize=8)
        point=ax.plot([],[],'bo',markersize=10)
        return line,point
    def anmi(i):
        ax.clear()
        line =ax.plot(w_1[0:i], w_2[0:i], Loss[0:i],'b:', markersize=8)
        point = ax.plot(w_1[i-1:i], w_2[i-1:i], Loss[i-1:i],'bo', markersize=10)
        return line,point
    anim = animation.FuncAnimation(fig, anmi, init_func=init,
                                   frames=1000, interval=10, blit=False,repeat=False)
    
    anim.save('GDtrajectory.gif', writer='imagemagick')
