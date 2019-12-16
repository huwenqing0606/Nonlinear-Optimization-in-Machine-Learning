from activations import Sigmoid, ReLU, Tanh, Exponential

class one_hidden_layer_network(object):
    
    def __init__(self, weight_a, weight_b, weight_c, layer_neuron_number, activation_name):
        self.weight_a=weight_a
        self.weight_b=weight_b
        self.weight_c=weight_c
        self.layer_neuron_number=layer_neuron_number
        self.activation_name=activation_name
    
    def output(self, x):
        y=0
        for j in range(self.layer_neuron_number):
            y=y+self.weight_c[j]*(self.activation_name.fn(self.activation_name, self.weight_a[j]*x-self.weight_b[j]))
        return y
    

    

        
