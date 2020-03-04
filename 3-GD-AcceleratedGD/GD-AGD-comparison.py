#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 20:57:02 2020

@author: Wenqing Hu (Missouri S&T)
"""

#comparison of convergence speed for GD and Nesterov for qudratic functions#

import numpy as np
import matplotlib.pyplot as plt

A=1
B=1000
epsilon=0.1

"""
The quadratic function f and its gradients
f(x_1, x_2)=0.5 A x_1^2 + 0.5 B x_2^2
"""
class function_f(object):
    def __init__(self,
                 axA=A,
                 axB=B,
                 name="f"):
        self.axA=axA
        self.axB=axB
        self.name=name
        
    def value(self, x_1, x_2):
        return 0.5*self.axA*x_1*x_1+0.5*self.axB*x_2*x_2
    
    def grad(self, x_1, x_2):
        return np.array([self.axA*x_1, self.axB*x_2])



"""
The perturbed quadratic function g and its gradients
g(x_1, x_2)=0.5 A x_1^2 + 0.5 B x_2^2+ epsilon(x_1^2+x_2^2)^{3/2}
"""
class function_g(object):
    def __init__(self,
                 axA=A,
                 axB=B,
                 eps=epsilon,
                 name="g"):
        self.axA=axA
        self.axB=axB
        self.eps=epsilon
        self.name=name
        
    def value(self, x_1, x_2):
        return 0.5*self.axA*x_1*x_1+0.5*self.axB*x_2*x_2+self.eps*((np.sqrt(x_1*x_1+x_2*x_2))**3)
    
    def grad(self, x_1, x_2):
        return np.array([self.axA*x_1+3*self.eps*x_1*np.sqrt(x_1**2+x_2**2), 
                         self.axB*x_2+3*self.eps*x_2*np.sqrt(x_1**2+x_2**2)])

    

"""
The non-convex function h and its gradients
h(x_1, x_2)=0.5 A x_1^2 - 0.5 B x_2^2
"""
class function_h(object):
    def __init__(self,
                 axA=A,
                 axB=B,
                 name="h"):
        self.axA=axA
        self.axB=axB
        self.name=name
        
    def value(self, x_1, x_2):
        return 0.5*self.axA*x_1*x_1-0.5*self.axB*x_2*x_2
    
    def grad(self, x_1, x_2):
        return np.array([self.axA*x_1, -self.axB*x_2])

    
    
"""
The optimizer update for: GD, Heavy-Ball, Nesterov
"""
class optimizer(object):
    def __init__(self, 
                 function=function_f()):
        self.function=function
        
    def GD(self, x_1, x_2, lr):
        grad=self.function.grad(x_1, x_2)
        update=-lr*grad
        return update
    
    def HeavyBall(self, x_1, x_2, x_1_old, x_2_old, alpha, beta):
        grad=self.function.grad(x_1, x_2)
        momentum=np.array([x_1-x_1_old, x_2-x_2_old])
        return -alpha*grad+beta*momentum
    
    def Nesterov(self, x_1, x_2, x_1_old, x_2_old, alpha, beta):
        grad=self.function.grad(x_1+beta*(x_1-x_1_old), x_2+beta*(x_2-x_2_old))
        momentum=np.array([x_1-x_1_old, x_2-x_2_old])
        return -alpha*grad+beta*momentum

    def update(self, x_1, x_2, x_1_old, x_2_old, lr, alpha, beta, optimizer):
        update=np.array([0,0])
        if optimizer=="GD":
            update=self.GD(x_1, x_2, lr)
        elif optimizer=="HeavyBall":
            update=self.HeavyBall(x_1, x_2, x_1_old, x_2_old, alpha, beta)
        elif optimizer=="Nesterov":
            update=self.Nesterov(x_1, x_2, x_1_old, x_2_old, alpha, beta)
        else:
            update=np.array([0,0])
        return update



m=min(A,B)
L=max(A,B)
kappa=np.sqrt(L/m)

lr=1/L
alpha=1/L
beta=(np.sqrt(kappa)-1)/(np.sqrt(kappa)+1)

if __name__ == "__main__":
    function=function_f()
    x_seed=np.random.uniform(-10, 10, size=2)

    #get the loss and distance to zero sequence for GD#
    optname="GD"
    x_current=x_seed
    x_current_minus1=x_current
    trajectory_x_1=[]
    trajectory_x_2=[]
    loss_GD=[]
    distance_GD=[]
    for i in range(1000):
        trajectory_x_1.append(x_current[0]) 
        trajectory_x_2.append(x_current[1])
        loss_GD.append(function.value(x_current[0], x_current[1]))
        distance_GD.append(np.sqrt(x_current[0]*x_current[0]+x_current[1]*x_current[1]))
        optimization=optimizer(function=function)
        x=x_current+optimization.update(x_current[0], x_current[1], x_current_minus1[0], x_current_minus1[1], lr, alpha, beta, optname)
        x_current_minus1=x_current
        x_current=x

    #get the loss and distance to zero sequence for Nesterov#
    optname="Nesterov"
    x_current=x_seed
    x_current_minus1=x_current
    trajectory_x_1=[]
    trajectory_x_2=[]
    loss_nesterov=[]
    distance_nesterov=[]
    for i in range(1000):
        trajectory_x_1.append(x_current[0]) 
        trajectory_x_2.append(x_current[1])
        loss_nesterov.append(function.value(x_current[0], x_current[1]))
        distance_nesterov.append(np.sqrt(x_current[0]*x_current[0]+x_current[1]*x_current[1]))
        optimization=optimizer(function=function)
        x=x_current+optimization.update(x_current[0], x_current[1], x_current_minus1[0], x_current_minus1[1], lr, alpha, beta, optname)
        x_current_minus1=x_current
        x_current=x

    #plot and compare the loss and distance to zero sequences for GD and Nesterov#
    plt.figure(figsize = (14,10))
    plt.plot(loss_GD, '-', color='r')
    plt.plot(loss_nesterov, '--', color='b')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('function loss for GD (red, solid) and Nesterov (blue, dashed)')
    plt.savefig('loss_GDvsNesterov.png')
    plt.show()  

    plt.figure(figsize = (14,10))
    plt.plot(distance_GD, '-', color='r')
    plt.plot(distance_nesterov, '--', color='b')
    plt.xlabel('iteration')
    plt.ylabel('distance to zero')
    plt.title('distance to zero for GD (red, solid) and Nesterov (blue, dashed)')
    plt.savefig('distance0_GDvsNesterov.png')
    plt.show()  
