#GD, Heavy-Ball and Nesterov for qudratic functions and perturbed quadratic functions#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

"""
The quadratic function f and its gradients
f(x_1, x_2)=0.5 A x_1^2 + 0.5 B x_2^2
"""
class function_f(object):
    
    def __init__(self,
                 A=1,
                 B=5):
        self.A=A
        self.B=B
        
    def value(self, x_1, x_2):
        return 0.5*self.A*x_1*x_1+0.5*self.B*x_2*x_2
    
    def grad(self, x_1, x_2):
        return np.array([self.A*x_1, self.B*x_2])


"""
The optimizer update for: GD, Heavy-Ball, Nesterov
"""
class optimizer_update(object):
    
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



#test and plot the trajectory#

lr=0.01
alpha=0.01
beta=1

if __name__ == "__main__":
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
        update=optimizer_update(function=function)
        x=x_current+update.GD(x_current[0], x_current[1], lr)
        x_current=x
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(trajectory_x_1, trajectory_x_2, loss, label='GD trajectory')
    ax.legend()
    plt.show()
    plt.plot(trajectory_x_1, trajectory_x_2)
    plt.show()
    plt.plot(loss)
    plt.show()    
    plt.plot(distance)
    plt.show()
    
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
        update=optimizer_update(function=function)
        x=x_current+update.HeavyBall(x_current[0], x_current[1], x_current_minus1[0], x_current_minus1[1], alpha, beta)
        x_current_minus1=x_current
        x_current=x
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(trajectory_x_1, trajectory_x_2, loss, label='Heavy-Ball trajectory')
    ax.legend()
    plt.show()
    plt.plot(trajectory_x_1, trajectory_x_2)
    plt.show()
    plt.plot(loss)
    plt.show()
    plt.plot(distance)
    plt.show()
    
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
        update=optimizer_update(function=function)
        x=x_current+update.Nesterov(x_current[0], x_current[1], x_current_minus1[0], x_current_minus1[1], alpha, beta)
        x_current_minus1=x_current
        x_current=x
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(trajectory_x_1, trajectory_x_2, loss, label='Nesterov trajectory')
    ax.legend()
    plt.show()
    plt.plot(trajectory_x_1, trajectory_x_2)
    plt.show()
    plt.plot(loss)
    plt.show()
    plt.plot(distance)
    plt.show()
