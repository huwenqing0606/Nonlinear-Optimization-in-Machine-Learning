import numpy as np

class Sigmoid(object):
    def __init__(self):
        """
        A logistic sigmoid activation function.
        """
        super().__init__()

    def fn(self, z):
        """
        Evaluate the logistic sigmoid, :math:`\sigma`, on the elements of input `z`.

        .. math::

            \sigma(x_i) = \\frac{1}{1 + e^{-x_i}}
        """
        return 1 / (1 + np.exp(-z))

    def grad(self, x):
        """
        Evaluate the first derivative of the logistic sigmoid on the elements of `x`.

        .. math::

            \\frac{\partial \sigma}{\partial x_i} = \sigma(x_i) (1 - \sigma(x_i))
        """
        fn_x = self.fn(self, x)
        return fn_x * (1 - fn_x)

    def grad2(self, x):
        """
        Evaluate the second derivative of the logistic sigmoid on the elements of `x`.

        .. math::

            \\frac{\partial^2 \sigma}{\partial x_i^2} =
                \\frac{\partial \sigma}{\partial x_i} (1 - 2 \sigma(x_i))
        """
        fn_x = self.fn(self, x)
        return fn_x * (1 - fn_x) * (1 - 2 * fn_x)


class ReLU(object):
    """
    A rectified linear activation function.

    Notes
    -----
    "ReLU units can be fragile during training and can "die". For example, a
    large gradient flowing through a ReLU neuron could cause the weights to
    update in such a way that the neuron will never activate on any datapoint
    again. If this happens, then the gradient flowing through the unit will
    forever be zero from that point on. That is, the ReLU units can
    irreversibly die during training since they can get knocked off the data
    manifold.

    For example, you may find that as much as 40% of your network can be "dead"
    (i.e. neurons that never activate across the entire training dataset) if
    the learning rate is set too high. With a proper setting of the learning
    rate this is less frequently an issue." [*]_

    References
    ----------
    .. [*] Karpathy, A. "CS231n: Convolutional neural networks for visual recognition".
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ReLU"

    def fn(self, z):
        """
        Evaulate the ReLU function on the elements of input `z`.

        .. math::

            \\text{ReLU}(z_i)
                &=  z_i \\ \\ \\ \\ &&\\text{if }z_i > 0 \\\\
                &=  0 \\ \\ \\ \\ &&\\text{otherwise}
        """
        return np.clip(z, 0, np.inf)

    def grad(self, x):
        """
        Evaulate the first derivative of the ReLU function on the elements of input `x`.

        .. math::

            \\frac{\partial \\text{ReLU}}{\partial x_i}
                &=  1 \\ \\ \\ \\ &&\\text{if }x_i > 0 \\\\
                &=  0   \\ \\ \\ \\ &&\\text{otherwise}
        """
        return (x > 0).astype(int)

    def grad2(self, x):
        """
        Evaulate the second derivative of the ReLU function on the elements of input `x`.

        .. math::

            \\frac{\partial^2 \\text{ReLU}}{\partial x_i^2}  =  0
        """
        return np.zeros_like(x)


class Tanh(object):
    def __init__(self):
        """
        A hyperbolic tangent activation function.
        """
        super().__init__()

    def __str__(self):
        return "Tanh"

    def fn(self, z):
        """
        Compute the tanh function on the elements of input `z`.
        """
        return np.tanh(z)

    def grad(self, x):
        """
        Evaluate the first derivative of the tanh function on the elements
        of input `x`.

        .. math::

            \\frac{\partial \\tanh}{\partial x_i}  =  1 - \\tanh(x)^2
        """
        return 1 - np.tanh(x) ** 2

    def grad2(self, x):
        """
        Evaluate the second derivative of the tanh function on the elements
        of input `x`.

        .. math::

            \\frac{\partial^2 \\tanh}{\partial x_i^2} =
                -2 \\tanh(x) \left(\\frac{\partial \\tanh}{\partial x_i}\\right)
        """
        tanh_x = np.tanh(x)
        return -2 * tanh_x * (1 - tanh_x ** 2)


class Exponential(object):
    def __init__(self):
        """
        An exponential (base e) activation function.
        """
        super().__init__()

    def __str__(self):
        return "Exponential"

    def fn(self, z):
        """Evaluate the activation function :math:`\\text{Exponential}(z_i) = e^{z_i}`."""
        return np.exp(z)

    def grad(self, x):
        """
        Evaluate the first derivative of the exponential activation on the elements
        of input `x`.

        .. math::

            \\frac{\partial \\text{Exponential}}{\partial x_i}  =  e^{x_i}
        """
        return np.exp(x)

    def grad2(self, x):
        """
        Evaluate the second derivative of the exponential activation on the elements
        of input `x`.

        .. math::

            \\frac{\partial^2 \\text{Exponential}}{\partial x_i^2}  =  e^{x_i}
        """
        return np.exp(x)





if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    dic={'Sigmoid': Sigmoid, 'ReLU': ReLU, 'Tanh': Tanh, 'Exponential': Exponential}
    namelist=['Sigmoid', 'ReLU', 'Tanh', 'Exponential']
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
        plt.show()
