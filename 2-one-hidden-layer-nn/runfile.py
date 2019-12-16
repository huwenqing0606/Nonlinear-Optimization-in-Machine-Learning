import numpy as np
import matplotlib.pyplot as plt

from activations import Sigmoid, ReLU, Tanh, Exponential
from network import one_hidden_layer_network

layer_neuron_number=100000

dic={'Sigmoid': Sigmoid, 'ReLU': ReLU, 'Tanh': Tanh, 'Exponential': Exponential}
namelist=['Sigmoid', 'ReLU', 'Tanh', 'Exponential']


def plot_network_output():
    for name in namelist:
        X = np.linspace(-5, 5, 100)
        Y = []
        network_output=one_hidden_layer_network(weight_a=np.random.randn(layer_neuron_number), 
                                                weight_b=np.random.randn(layer_neuron_number), 
                                                weight_c=np.random.randn(layer_neuron_number), 
                                                layer_neuron_number=layer_neuron_number, 
                                                activation_name=dic[name])
        for i in range(100):
            Y.append(network_output.output(X[i]))
            
        plt.plot(X, Y)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('One hidden layer neural network'+' with '+name+' activation'+' hidden layer size='+str(layer_neuron_number))
        plt.legend()    
        plt.savefig('OneHiddenLayerNN_'+name+'_layersize='+str(layer_neuron_number)+'.jpg', bbox_inches='tight')
        plt.show()
        plt.close()

if __name__ == "__main__":
    plot_network_output()
