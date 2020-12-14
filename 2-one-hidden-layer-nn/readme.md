<b>Nonlinear Optimization in Machine Learning.</b>

2. The shape and loss landscape of output function for one hidden layer funnly connected neural networks.

Consider a neural network with one hidden layer that consists of $p$ neurons and input $x\in \mathbb{R}^1$, output $y\in \mathbb{R}^1$. The neural network function has the form 
$y(x)=\sum\limits_{j=1}^p c_j \sigma(a_j x- b_j) \ , $
where $\mathbf{a}=a_j, \mathbf{b}=b_j$ and $\mathbf{c}=c_j$ are the neural network weights. Assume $(a_j, b_j, c_j)\sim \mathcal{N}(0, I_3)$, $j=1,2,…,p$ is a family of i.i.d multivariate normal distributions. For different realizations of $(a_j, b_j, c_j)$, plot the function $y(x)$ on $x-y$ graph. Experiment different hidden layer sizes.

With the above done, assume that the training data $(x, y)$ follows a bivariate normal distribution $\mathcal{N}(0, I_2)$. Let the variables $a_1$ and $a_2$ vary in a certain interval. Let all other neural network weights $a_j, b_j, c_j$ follow the same assumption as above. For different $(a_1, a_2)$, plot the empirical loss function of this network as a function of $(a_1, a_2)$. Is the loss function convex with respect to $(a_1, a_2)$? Experiment different hidden layer sizes.
