<b>MATH6001-106. Nonlinear Optimization in Machine Learning.</b>

3. Gradient Descent v.s. Accelarated Gradient Descent on quadratic and perturbed quadratic functions.

Consider a two-dimensional quadratic function $f(x_1, x_2) = 0.5 A x_1^2 + 0.5 B x_2^2$ for $A, B>0$ and a small perturbation of this function $g(x_1, x_2) = 0.5 A x_1^2 + 0.5 B x_2^2 + \epsilon (x_1^2+x_2^2)^{3/2}$ for $\epsilon>0$. Use Gradient Descent, Polyak's heavy ball method as well as Nesterov's Acclerated Gradient Descent to optimize $f$ and $g$. Experiment (1) different/random initializations; (2) directly calculating the gradient as well as using tensorflow gradient for a general gradient calculation; (3) plot the trajectory as well as show its dynamics via an animation; (4) plot the evolution of the error function as a function of number of iterations; (5) compare on different parmeters $A, B, \epsilon>0$.

Perform the same experiments for the non-convex function $h(x_1, x_2) = 0.5 A x_1^2 - 0.5 B x_2^2$.
