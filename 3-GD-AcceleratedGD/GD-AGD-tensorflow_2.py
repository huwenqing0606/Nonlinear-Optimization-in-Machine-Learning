#GD, Heavy-Ball and Nesterov for qudratic functions and perturbed quadratic functions#
#automatic differentiation in tensorflow#
#developed under tensorflow v1.14.0#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

import tensorflow as tf
tf.enable_eager_execution() #tf.placeholder is not allowed in eager_execution mode#


A=1
B=1
epsilon=1



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
        
    def value(self, tfx_1, tfx_2):
        #tfx_1 and tfx_2 must be two tensorflow variables#
        y=tf.add(tf.multiply(0.5*self.axA, tf.square(tfx_1)), tf.multiply(0.5*self.axB, tf.square(tfx_2)))
        #output is a tensorflow variable, for actual value, must use y.numpy()#
        return y



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
        
    def value(self, tfx_1, tfx_2):
        s=tf.add(tf.multiply(0.5*self.axA, tf.square(tfx_1)), tf.multiply(0.5*self.axB, tf.square(tfx_2)))
        y=tf.add(s, tf.multiply(self.eps, tf.pow(tf.sqrt(tf.add(tf.square(tfx_1), tf.square(tfx_2))), 3)))
        return y
    


"""
The non-convex function h and its gradients
h(x_1, x_2)=0.5 A x_1^2 - 0.5 B x_2^2
the gradients are calculated via the tf.GradientTape() mode
"""
class function_h(object):
    def __init__(self,
                 axA=A,
                 axB=B,
                 name="h"):
        self.axA=axA
        self.axB=axB
        self.name=name
        
    def value(self, tfx_1, tfx_2):
        y=tf.subtract(tf.multiply(0.5*self.axA, tf.square(tfx_1)), tf.multiply(0.5*self.axB, tf.square(tfx_2)))
        return y
   
    

"""
calculate the gradients using tf.GradientTape()
tfx_1 and tfx_2 must be tensorflow variables
"""
class grad(object):
    def __init__(self,
                 function=function_f()):
        self.function=function
    
    def calculate(self, tfx_1, tfx_2):
        with tf.GradientTape() as tape:
            y=function.value(tfx_1, tfx_2)
        grad_x_1, grad_x_2=tape.gradient(y, [tfx_1, tfx_2])
        return np.array([grad_x_1.numpy(), grad_x_2.numpy()])
    


"""
The optimizer update for: GD, Heavy-Ball, Nesterov
"""
class optimizer(object):
    def __init__(self, 
                 function=function_f()):
        self.function=function
        
    def GD(self, x_1, x_2, lr):
        tfx_1=tf.Variable(initial_value=x_1, dtype='float')
        tfx_2=tf.Variable(initial_value=x_2, dtype='float')
        gradient=grad(function=self.function)
        grd=gradient.calculate(tfx_1, tfx_2)
        update=-lr*grd
        return update
    
    def HeavyBall(self, x_1, x_2, x_1_old, x_2_old, alpha, beta):
        tfx_1=tf.Variable(initial_value=x_1, dtype='float')
        tfx_2=tf.Variable(initial_value=x_2, dtype='float')
        gradient=grad(function=self.function)
        grd=gradient.calculate(tfx_1, tfx_2)
        momentum=np.array([x_1-x_1_old, x_2-x_2_old])
        return -alpha*grd+beta*momentum
    
    def Nesterov(self, x_1, x_2, x_1_old, x_2_old, alpha, beta):
        tfx_1=tf.Variable(initial_value=x_1+beta*(x_1-x_1_old), dtype='float')
        tfx_2=tf.Variable(initial_value=x_2+beta*(x_2-x_2_old), dtype='float')
        gradient=grad(function=self.function)
        grd=gradient.calculate(tfx_1, tfx_2)
        momentum=np.array([x_1-x_1_old, x_2-x_2_old])
        return -alpha*grd+beta*momentum

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

lr=0.01
alpha=0.01
beta=1

if __name__ == "__main__":
    for optname in {"GD", "HeavyBall", "Nesterov"}:
        x_current=np.random.uniform(-1, 1, size=2)
        x_current_minus1=x_current
        trajectory_x_1=[]
        trajectory_x_2=[]
        loss=[]
        distance=[]
        function=function_g()
        for i in range(1000):
            trajectory_x_1.append(x_current[0]) 
            trajectory_x_2.append(x_current[1])
            tfx_1=tf.Variable(initial_value=x_current[0], dtype='float')
            tfx_2=tf.Variable(initial_value=x_current[1], dtype='float')
            loss.append(function.value(tfx_1, tfx_2).numpy())
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

