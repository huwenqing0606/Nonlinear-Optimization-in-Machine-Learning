<b>MATH6001 Nonlinear Optimization in Machine Learning</b>

5. SGD and Variance-Reduced SGD

Consider a two-dimensional quadratic loss function $L(w_1, w_2; (x_1, x_2, y)) = 0.5(Aw_1x_1+Bw_2x_2-y)^2$ for $A, B>0$. Let the training data (x_1, x_2, y) be a size $n$ sample of i.i.d normal distributions $\mathcal{N}(0, I_3)$. Use (a) Stochastic Gradient Descent (SGD); (b) Stochastic Variance-Reduced Gradient (SVRG); (c) StochAstic Recursive grAdient algoritHm (SARAH) to optimize the empirical loss function. You can calculate the gradients using the automatic differentiation in tensorflow.

Experiment (1) different/random initializations; (2) plot the trajectory as well as show its dynamics via an animation; (3) plot the evolution of the training error (loss) as a function of number of iterations; (4) plot the evolution of the test error as a function of number of iterations; (5) compare on different parmeters $A, B$; (6) compare for different training sample size and different batchsizes, as well as different epoch lengthes for SVRG and SARAH.

