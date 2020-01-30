# -*- coding: utf-8 -*- #
"""
Created on Wed Jan 22 08:23:40 2020
@author: Wenqing Hu (Missouri S&T)
"""

"""
Construct the output function of a fully connnected neural network 
with L layers and network size n_1, ..., n_L
parameters L, n_1, ..., n_L are given
"""

import numpy as np
from activations import Sigmoid, ReLU, Tanh, Exponential

outputfile = open('fullnetworkoutput_ReLU.txt', 'w') 

"""
one layer of the neural network, input vector x^{in}, output \sigma(W x^{in} + b)
W is weight vector, b is the bias
"""
class onelayer(object):
    
    def __init__(self,
                 inputvector=[],    #input vector， has to be array type#
                 activation=Sigmoid(),        #activation function, will be applied termwise#
                 weight=[],         #weights from input to output layer, should be an array of size outputsize x inputsize#
                 bias=[],           #bias vectors in the particular layer, a vector array of length = outputsize#
                 ):
        self.inputvector=inputvector
        self.activation=activation
        self.weight=weight
        self.bias=bias
        
    def output(self):
        print("weight*input=\n",np.dot(self.weight,self.inputvector), file=outputfile)
        preoutput=np.dot(self.weight,self.inputvector)+self.bias
        print("preoutput=\n", preoutput, file=outputfile)
        length=preoutput.size
        output=[]
        for i in range(length):
            output.append(self.activation.fn(preoutput[i]))
        print("output=\n", np.array(output), file=outputfile)
        return np.array(output)
    



"""
a fully connected neural network with L hidden layers, input is a number x, output is a number y
L hidden layers with layer sizes n_1, ..., n_L
activation are given the same for all layers
all weights and biases are initialized under the LeCun initilization: W_{ij} as N(0,1/n_l) where l is the label of hidden layer and b_k as N(0,1)
"""    
class fullnetwork(object):
    
    def __init__(self,
                 L=1, #number of hidden layers#
                 n=np.random.randint(1, 6, size=1), #network size for each hidden layer n[0]=n_1, ..., m[L-1]=n_L#
                 activation=Sigmoid()):
        self.L=L
        self.n=n
        self.activation=activation
    
    def setparameter(self):
        #initialize the weights and the biases according to the LeCun initialization#
        weight=[]
        bias=[]
        #the initial layer#
        weight.append(np.array([[np.random.normal(loc=0.0, scale=1.0) for i in range(1)] for j in range(self.n[0])]))
        bias.append(np.array([[np.random.normal(loc=0.0, scale=1.0) for i in range(1)] for j in range(self.n[0])]))
        for l in range(self.L):
            if l==self.L-1:
                #the last layer#
                weight.append(np.array([[np.random.normal(loc=0.0, scale=1/np.sqrt(self.n[l])) for i in range(self.n[l])] for j in range(1)]))
                bias.append(np.array([[np.random.normal(loc=0.0, scale=1.0) for i in range(1)] for j in range(1)]))                        
            else:
                #all hidden layers except the last layer#
                weight.append(np.array([[np.random.normal(loc=0.0, scale=1/np.sqrt(self.n[l])) for i in range(self.n[l])] for j in range(self.n[l+1])]))
                bias.append(np.array([[np.random.normal(loc=0.0, scale=1.0) for i in range(1)] for j in range(self.n[l+1])]))                        
        return weight, bias
    
    def output(self, x, weight, bias):
        layervector=np.array(x) #layervector corresponds to the outputs of all neurons at the current layer#
        for l in range(self.L+1):
            #all layers including the initial and the last layer#
            print("***********************layer ", l, "***********************", file=outputfile)
            print("layervector=\n", layervector, file=outputfile)
            print("weight=\n", np.array(weight[l]), file=outputfile)
            print("bias=\n", np.array(bias[l]), file=outputfile)
            addlayer=onelayer(inputvector=layervector, activation=self.activation, weight=np.array(weight[l]), bias=np.array(bias[l]))
            layervector=addlayer.output()
        return layervector
    
    

"""
test the output
"""
if __name__ == "__main__":
    L=10 #number of hidden layers#
    n=np.random.randint(1, 10, size=L) #network size for each hidden layer n[0]=n_1, ..., m[L-1]=n_L#
    print("hidden layer sizes=", n, file=outputfile)
    network=fullnetwork(L=L, n=n, activation=ReLU())
    weight, bias=network.setparameter()
    print("\nnetwork output=", float(network.output(1, weight, bias)), file=outputfile)
    
    outputfile.close() 