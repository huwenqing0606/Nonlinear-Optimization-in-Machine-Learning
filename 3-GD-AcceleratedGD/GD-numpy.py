def f(x):
    return x[0] + 2 * x[1] + 4

def error(x):
    return (f(x) - 0)**2

def gradient_descent(x):
    delta = 0.00000001
    derivative_x0 = (error([x[0] + delta, x[1]]) - error([x[0], x[1]])) / delta
    derivative_x1 = (error([x[0], x[1] + delta]) - error([x[0], x[1]])) / delta
    rate = 0.02
    x[0] = x[0] - rate * derivative_x0
    x[1] = x[1] - rate * derivative_x1
    return [x[0], x[1]]

x = [-0.5, -1.0]
for i in range(100):
    x = gradient_descent(x)
    print('x = {:6f},{:6f}, f(x) = {:6f}'.format(x[0],x[1],f(x)))
