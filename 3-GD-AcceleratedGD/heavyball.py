#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:21:01 2020

@author: UM-AD\huwen
"""


#Heavy-Ball for qudratic functions and perturbed quadratic functions#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

A=1
B=2
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



#test and plot the trajectory#

m=min(A,B)
L=max(A,B)
kappa=np.sqrt(L/m)

lr=1/L
alpha=4/(np.sqrt(L)+np.sqrt(m))**2
beta=(np.sqrt(kappa)-1)/(np.sqrt(kappa)+1)

if __name__ == "__main__":
    for optname in {"HeavyBall"}:
        x_current=np.random.uniform(-1, 1, size=2)
        x_current_minus1=x_current
        trajectory_x_1=[]
        trajectory_x_2=[]
        loss=[]
        distance=[]
        function=function_f()
        for i in range(1000):
            trajectory_x_1.append(x_current[0]) 
            trajectory_x_2.append(x_current[1])
            loss.append(function.value(x_current[0], x_current[1]))
            distance.append(np.sqrt(x_current[0]*x_current[0]+x_current[1]*x_current[1]))
            optimization=optimizer(function=function)
            x=x_current+optimization.update(x_current[0], x_current[1], x_current_minus1[0], x_current_minus1[1], lr, alpha, beta, optname)
            x_current_minus1=x_current
            x_current=x

        fig = plt.figure()
        ax = Axes3D(fig)
        u = np.linspace(-10, 10, 100)
        v = np.linspace(-10, 10, 100)
        u=np.array(u)
        v=np.array(v)
        u, v = np.meshgrid(u, v)
        w=function.value(u,v)
        ax.plot_surface(u, v, w, rstride=1, cstride=1, cmap='summer')
        ax.set_xlabel('x_1')
        ax.set_ylabel('x_2')
        ax.set_zlabel(function.name+'(x_1, x_2)')
        plt.savefig('Landscape_'+optname+'_A='+str(A)+'_B='+str(B)+'_alpha='+str(alpha)+'_beta='+str(beta)+'_eps='+str(epsilon)+'.jpg')
        plt.show()

        fig = plt.figure()
        mpl.rcParams['legend.fontsize'] = 10
        ax = fig.gca(projection='3d')
        ax.plot(trajectory_x_1, trajectory_x_2, loss, label=optname+' trajectory', color='r')
        ax.legend()
        ax.set_xlabel('x_1')
        ax.set_ylabel('x_2')
        ax.set_zlabel(function.name+'(x_1, x_2)') 
        plt.show()

        plt.plot(trajectory_x_1, trajectory_x_2)
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.title(optname)
        plt.show()

        plt.plot(loss)
        plt.xlabel('iteration')
        plt.ylabel('function error to minimum')
        plt.title(optname)
        plt.savefig('Loss_'+optname+'_A='+str(A)+'_B='+str(B)+'_alpha='+str(alpha)+'_beta='+str(beta)+'_eps='+str(epsilon)+'.jpg')
        plt.show()

        plt.plot(distance)
        plt.xlabel('iteration')
        plt.ylabel('distance to minimizer')
        plt.title(optname)
        plt.savefig('Distance_To_Zero_'+optname+'_A='+str(A)+'_B='+str(B)+'_alpha='+str(alpha)+'_beta='+str(beta)+'_eps='+str(epsilon)+'.jpg')
        plt.show()
        
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
            line =ax.plot(trajectory_x_1[0:i],trajectory_x_2[0:i],loss[0:i],'b:', markersize=8)
            point = ax.plot(trajectory_x_1[i-1:i],trajectory_x_2[i-1:i],loss[i-1:i],'bo', markersize=10)
            return line,point
        anim = animation.FuncAnimation(fig, anmi, init_func=init,
                                       frames=1000, interval=10, blit=False,repeat=False)
        anim.save(optname+'_A='+str(A)+'_B='+str(B)+'_alpha='+str(alpha)+'_beta='+str(beta)+'_eps='+str(epsilon)+'.gif', writer='imagemagick')

