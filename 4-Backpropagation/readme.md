<b>MATH6001-106. Nonlinear Optimization in Machine Learning.</b>

4. Backpropagation algorithm

Consider an $L$-hidden-layer fully connected neural network with hidden layer sizes $n_1, ..., n_L$, where the input layer has only one neuron $x\in \mathbb{R}$ and output layer has only one neuron $y\in \mathbb{R}$. Let the trainig data $(x_1, y_1), ..., (x_n, y_n)$ follow a bivariate unit normal distribution $(x, y)\sim \mathcal{N}(0, I_2)$. Use the backpropagation algorithm, calculate the gradient vector of the empirical loss function and the population loss function of the neural network for a quadratic loss. Experiments on (1) different hidden layer sizes ($n_1, ..., n_L$); (2) different number of hidden layers ($L$); (3) different sizes of the training data set ($n$).

Following the above, plot the loss function and the trajectory of the gradient descent algorithm by picking two network parameters and randomly choosing other network parameters as i.i.d. unit normal distributions.
