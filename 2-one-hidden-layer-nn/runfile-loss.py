import numpy as np
import matplotlib.pyplot as plt

from activations import Sigmoid, ReLU, Tanh, Exponential
from network import one_hidden_layer_network
from mpl_toolkits.mplot3d import Axes3D

layer_neuron_number=10000
training_size=1
N=100

dic={'Sigmoid': Sigmoid, 'ReLU': ReLU, 'Tanh': Tanh, 'Exponential': Exponential}
namelist=['Sigmoid', 'ReLU', 'Tanh', 'Exponential']

weight_a_secondpart=np.random.randn(layer_neuron_number-2)
weight_b=np.random.randn(layer_neuron_number) 
weight_c=np.random.randn(layer_neuron_number)

X=[]
for n in range(training_size):
    X.append(np.random.normal(0,1,1))

Y=[]
for n in range(training_size):
    Y.append(np.random.normal(0,1,1))

def plot_network_loss():
    for name in namelist:
        a_1 = np.linspace(-10, 10, N)
        a_2 = np.linspace(-10, 10, N)
        L = [[0 for i in range(N)] for j in range(N)]
        for i in range(N):
            for j in range (N):
                weight_a=[]
                weight_a.append(a_1[i])
                weight_a.append(a_2[j])
                for k in range(layer_neuron_number-2):
                    weight_a.append(weight_a_secondpart[k])
                network_output=one_hidden_layer_network(weight_a=weight_a, 
                                                        weight_b=weight_b,
                                                        weight_c=weight_c,
                                                        layer_neuron_number=layer_neuron_number, 
                                                        activation_name=dic[name])
                Z=[]
                for n in range(training_size):
                    Z.append((Y[n]-network_output.output(X[n]))**2)                
                L[i][j]=0.5*np.mean(np.array(Z))
                              
        fig = plt.figure()
        ax = Axes3D(fig)
        u=np.array(a_1)
        v=np.array(a_2)
        w=np.array(L)
        u, v = np.meshgrid(u, v)
        ax.plot_surface(u, v, w, rstride=1, cstride=1, cmap='rainbow')
        ax.set_title(name+" empirical loss landscape, hidden layer neuron number="+str(layer_neuron_number)+", training size="+str(training_size))
        ax.set_zlabel('Empirical Loss') 
        ax.set_xlabel('weight a_1')
        ax.set_ylabel('weight a_2')
        plt.legend()    
        plt.savefig('OneHiddenLayerNN-Loss_'+name+'_layersize='+str(layer_neuron_number)+", training size="+str(training_size)+'.jpg', bbox_inches='tight')
        plt.show()
        plt.close()

if __name__ == "__main__":
    plot_network_loss()