import numpy as np
import matplotlib.pyplot as plt

from activations import Sigmoid, ReLU, Tanh, Exponential

dic={'Sigmoid': Sigmoid, 'ReLU': ReLU, 'Tanh': Tanh, 'Exponential': Exponential}
namelist=['Sigmoid', 'ReLU', 'Tanh', 'Exponential']

def plot_activations():
    for name in namelist:
        X = np.linspace(-5, 5, 100)
        Y = []
        Ygrad=[]
        Ygrad2=[]
        for i in range(100):
            Y.append(dic[name].fn(dic[name], X[i]))
            Ygrad.append(dic[name].grad(dic[name], X[i]))
            Ygrad2.append(dic[name].grad2(dic[name], X[i]))
            
        plt.plot(X, Y, label=r"$y$")
        plt.plot(X, Ygrad, label=r"$\frac{dy}{dx}$")
        plt.plot(X, Ygrad2, label=r"$\frac{d^2 y}{dx^2}$")

        plt.xlabel('x')
        plt.ylabel('y='+name+'(x)')
        plt.legend()    
        plt.savefig('activations_'+name+'.pdf', bbox_inches='tight')
        plt.show()
        plt.close()

if __name__ == "__main__":
    plot_activations()
