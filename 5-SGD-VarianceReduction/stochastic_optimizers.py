#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:44:22 2020

@author: Wenqing Hu (Missouri S&T)
"""

#SGD, SVRG and SARAH for quadratic loss and Gaussian input data
#tensorflow version=1.14.0

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from random import sample

import tensorflow as tf
tf.enable_eager_execution()


A=1
B=1
training_sample_size=100
batchsize=75
num_steps=1000
lr=0.01

"""
Loss Function L(w_1, w_2; (x_1, x_2, y)) = 0.5(Aw_1x_1+Bw_2x_2-y)^2 for A, B>0
and its gradients with respect to the weight parameters w_1 and w_2
the gradients are calculated via the tf.GradientTape() mode
"""
class LossFunction(object):
    def __init__(self,
                 axA=A,
                 axB=B):
        self.axA=axA
        self.axB=axB
    
    #value of the loss function    
    def value(self, w, x, y):
        return 0.5*(self.axA*w[0]*x[0]+self.axB*w[1]*x[1]-y)**2
    
    #gradient of the loss function with respect to the weights (w_1, w_2)
    def grad(self, w, x, y):
        tfw_1=tf.Variable(initial_value=w[0], dtype='float')
        tfw_2=tf.Variable(initial_value=w[1], dtype='float')
        with tf.GradientTape() as tape:
            loss=0.5*tf.math.square(tf.subtract(tf.add(tf.multiply(tf.multiply(self.axA, tfw_1), x[0]), tf.multiply(tf.multiply(self.axB, tfw_2), x[1])), y))
        grad_w_1, grad_w_2=tape.gradient(loss, [tfw_1, tfw_2])
        return np.array([grad_w_1.numpy(), grad_w_2.numpy()])
    
    #average of a sequence of loss functions for a given list of training samples (x_i, y_i)
    def average(self, w, training_sample_x, training_sample_y, function):
        average=function(w, training_sample_x[0], training_sample_y[0])
        size=len(training_sample_y)
        for i in range(size-1):
            average+=function(w, training_sample_x[i+1], training_sample_y[i+1])
        average=average/size
        return average
   
    
"""
The stochastic optimizer update for: SGD, SVRG, SARAH
"""
class stochastic_optimizer(object):
    def __init__(self, 
                 function=LossFunction()):
        self.function=function
        
    def SGD(self, w, training_sample_x, training_sample_y, lr, batchsize):
        batch_x=sample(list(training_sample_x), batchsize)
        batch_y=sample(list(training_sample_y), batchsize)
        batch_x=np.array(batch_x)
        batch_y=np.array(batch_y)
        grad=self.function.average(w, batch_x, batch_y, self.function.grad)
        update=-lr*grad
        return update
    
    def update(self, w, training_sample_x, training_sample_y, lr, batchsize, optname):
        update=np.array([0,0])
        if optname=="SGD":
            update=self.SGD(w, training_sample_x, training_sample_y, lr, batchsize)
        else:
            update=np.array([0,0])
        return update
        
        
"""
running the code, plot 
(1) the trajectory animation; 
(2) the evolution of the training error; 
(3) the evolution of generalization error.
"""
if __name__ == "__main__":
    #generate the training samples (x_i, y_i)#
    training_sample_x=np.random.normal(0,1,size=(training_sample_size, 2))
    training_sample_y=np.random.normal(0,1,size=training_sample_size)
    #initialize the initial weights
    w_init=np.random.uniform(-1, 1, size=2)
    #optimization step obtain a sequence of losses and weights trajectory
    for optname in {"SGD"}:
        w_current=w_init
        w_current_minus1=w_current
        trajectory_w_1=[]
        trajectory_w_2=[]
        loss_list=[]
        function=LossFunction()
        for i in range(num_steps):
            trajectory_w_1.append(w_current[0]) 
            trajectory_w_2.append(w_current[1])
            loss_list.append(function.average(w_current, training_sample_x, training_sample_y, function.value))
            stochastic_optimization=stochastic_optimizer(function=function)
            w=w_current+stochastic_optimization.update(w_current, training_sample_x, training_sample_y, lr, batchsize, optname)
            w_current_minus1=w_current
            w_current=w

        #plot the trajctory as an animation
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
            line =ax.plot(trajectory_w_1[0:i],trajectory_w_2[0:i],loss_list[0:i],'b:', markersize=8)
            point = ax.plot(trajectory_w_1[i-1:i],trajectory_w_2[i-1:i],loss_list[i-1:i],'bo', markersize=10)
            return line,point
        anim = animation.FuncAnimation(fig, anmi, init_func=init,
                                       frames=num_steps, interval=10, blit=False,repeat=False)
        anim.save(optname+'_A='+str(A)+'_B='+str(B)+'_trainingsize='+str(training_sample_size)+'_batchsize='+str(batchsize)+'_learningrate='+str(lr)+'_steps='+str(num_steps)+'.gif', writer='imagemagick')
