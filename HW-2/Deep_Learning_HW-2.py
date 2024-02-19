import numpy as np

# Calculates the output of this NN ReLU-ReLU-ReLU
# Returns output of nodes: a3 (rightmost-y), a2 (bottom), a1 (top)
def rrr_NN(x1, x2, w1, w2, w3, w4, w5, w6):
    a1 = max(0.0, x1*w1+x2*w3)
    a2 = max(0.0, x1*w2+x2*w4)
    a3 = max(0.0, a1*w5+a2*w6)
    return a3, a2, a1


# Calculates the output of this NN ReLU-ReLU-
# Returns output of nodes: a3 (rightmost-y), a2 (bottom), a1 (top)
def rrt_NN(x1, x2, w1, w2, w3, w4, w5, w6):
    a1 = max(0.0, x1*w1+x2*w3)
    a2 = max(0.0, x1*w2+x2*w4)
    a3 = np.tanh(a1 * w5 + a2 * w6)
    return a3, a2, a1


# Calculates the Loss Function
def loss_NN(X, y, w1, w2, w3, w4, w5, w6):
    loss = 0.0
    i = 0;
    for x in X:
        # Get the inputs
        x1 = x[0]
        x2 = x[1]
        y0 = y[i]
        i += 1
        # Calculate the output of all the nodes
        a3, a2, a1 = rrr_NN(x1, x2, w1, w2, w3, w4, w5, w6)
        loss = loss + (a3 - y0)**2
    loss = loss / len(X)
    return loss


# Calculates the gradient of the Loss Function for the ReLU-ReLU-ReLU network
# Returns a vector of the partial derivative values [∂L/∂w1, ∂L/∂w2, ∂L/∂w3, ∂L/∂w4, ∂L/∂w5, ∂L/∂w6]
def gradient_rrr_NN(X, y, w1, w2, w3, w4, w5, w6):
    g = np.zeros(6)
    i = 0;
    for x in X:
        # Get the inputs
        x1 = x[0]
        x2 = x[1]
        y0 = y[i]
        i += 1
        # Calculate the output of all the nodes
        a3, a2, a1 = rrr_NN(x1, x2, w1, w2, w3, w4, w5, w6)
        # Sum the gradient components
        g[0] = g[0] + 2*(a3-y0)*np.heaviside(a3, 1)*np.heaviside(a1, 1)*w5*x1
        g[1] = g[1] + 2*(a3-y0)*np.heaviside(a3, 1)*np.heaviside(a2, 1)*w6*x1
        g[2] = g[2] + 2*(a3-y0)*np.heaviside(a3, 1)*np.heaviside(a1, 1)*w5*x2
        g[3] = g[3] + 2*(a3-y0)*np.heaviside(a3, 1)*np.heaviside(a2, 1)*w6*x1
        g[4] = g[4] + 2*(a3-y0)*np.heaviside(a3, 1)*a1
        g[5] = g[5] + 2*(a3-y0)*np.heaviside(a3, 1)*a2
    for i in range(6):
        g[i] = g[i] / len(X)
    return g


# Calculates the gradient of the Loss Function for the ReLU-ReLU-tanh network
# Returns a vector of the partial derivative values [∂L/∂w1, ∂L/∂w2, ∂L/∂w3, ∂L/∂w4, ∂L/∂w5, ∂L/∂w6]
def gradient_rrt_NN(X, y, w1, w2, w3, w4, w5, w6):
    g = np.zeros(6)
    i = 0;
    for x in X:
        # Get the inputs
        x1 = x[0]
        x2 = x[1]
        y0 = y[i]
        i += 1
        # Calculate the output of all the nodes
        a3, a2, a1 = rrt_NN(x1, x2, w1, w2, w3, w4, w5, w6)
        # Calculate the tanh'(a3)
        tanhd_a3 = 1 - np.tanh(a3)**2
        # Sum the gradient components
        g[0] = g[0] + 2*(a3-y0)*tanhd_a3*np.heaviside(a1, 1)*w5*x1
        g[1] = g[1] + 2*(a3-y0)*tanhd_a3*np.heaviside(a2, 1)*w6*x1
        g[2] = g[2] + 2*(a3-y0)*tanhd_a3*np.heaviside(a1, 1)*w5*x2
        g[3] = g[3] + 2*(a3-y0)*tanhd_a3*np.heaviside(a2, 1)*w6*x1
        g[4] = g[4] + 2*(a3-y0)*tanhd_a3*a1
        g[5] = g[5] + 2*(a3-y0)*tanhd_a3*a2
    for i in range(6):
        g[i] = g[i] / len(X)
    return g

if __name__ == '__main__':
    # The initial Dataset:
    X = [[-1.0, 1.0],
         [0.5, 0.5],
         [1.0, -1.0],
         [0.0, 0.5]]
    y = [0.0, 1.0, 1.0, 0.0]
    # The initial weights
    w1 = 0.05
    w2 = 0.03
    w3 = 0.02
    w4 = 0.01
    w5 = 0.5
    w6 = 1.0
    # Question 1
    # a. (30 pts): Using the Back-Propagation algorithm, compute the gradient
    # (a vector of 6 components) of the loss function L(w;X,y)=1/|X|  ∑_((x,y)∈(X,y))(net(x;w)-y)^2
    # where the function net(x;w) is the output of the neural network when the input is the
    # vector x, and all nodes in the network use as activation function the ReLU function
    # a(x)=x_+=max(x,0).
    g = gradient_rrr_NN(X, y, w1, w2, w3, w4, w5, w6)
    print("\nQuestion1.a:")
    print("Gradient: ", g)
    # Question 1
    # b. (10 pts) Check your answer in step a. by computing the difference quotient
    # (L(w+he_1;X,y)-L(w;X,y))/h where h=0.1,0.01,0.001 and e_1=[1,0,0,0,0,0]^T
    # (the unit vector in the first dimension).
    # What is the different between your computed value of ∂L/∂w_1 and the
    # difference quotient for the three proposed values of h?
    print("\nQuestion1.b:")
    loss = loss_NN(X, y, w1, w2, w3, w4, w5, w6)
    for h in [0.1, 0.01, 0.001]:
        w_1 = w1+h
        lossh = loss_NN(X, y, w_1, w2, w3, w4, w5, w6)
        quotient = (lossh-loss)/h
        dL_dw1 = g[0]
        print(f"h: {h:.3f}\tDifference Quotient: ", quotient, "\t∂L/∂w_1:", dL_dw1)
    print("As h becomes smaller and smaller the Difference Quotient -> ∂L/∂w_1.")

    # Question 2 (25 pts): Using a step-size h=0.001, use the GD equation:
    # w'←w-h∇L(w;X,y)
    # to compute a new set of weights w' given the gradient computed at Question 1.a.
    # Now, compute the value of the loss function L(w^';X,y) of the neural network
    # in the figure above, evaluated at the point w'.
    # How does the new loss value compare to the original value L(w;X,y) ?
    h = 0.001
    w_1 = w1-0.001*g[0]
    w_2 = w2-0.001*g[1]
    w_3 = w3-0.001*g[2]
    w_4 = w4-0.001*g[3]
    w_5 = w5-0.001*g[4]
    w_6 = w6-0.001*g[5]
    loss = loss_NN(X, y, w1, w2, w3, w4, w5, w6)
    lossh = loss_NN(X, y, w_1, w_2, w_3, w_4, w_5, w_6)
    print("\nQuestion2:")
    print("Original loss (w,X,y):", loss, "Updated loss (w',X,y):", lossh)
    print("The loss is slightly lower with the updated weights, as expected")

    # Question 3 (35 pts): For the network in the figure above, compute the gradient function of
    # the single-instance mini-batch:
    # x1	x2	y (label)
    # -1	1	0
    # The activation function of the two hidden nodes is the rectified linear unit
    # (ReLU): a(x)=x_+=max{x,0} and the activation function for the output node is
    # the hyperbolic tangent function a(x)=tanhx.
    # The loss function whose gradient we seek is again the same as in Question 1
    # (mean square error)
    print("\nQuestion2:")
    #  Calculate the gradient
    g = gradient_rrt_NN([[-1.0, 1.0]], [0], w1, w2, w3, w4, w5, w6)
    print("Gradient: ", g)
    print("Not a good mini batch ;-)")
