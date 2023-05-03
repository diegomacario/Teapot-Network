# neural nets are just mathematical expressions

# leaf nodes of graph are weights of NN and inputs

# we need gradients of weights and biases but not of inputs, since the inputs are a given

# if you want L to go up, you move weights along the direction of their gradients
# if you want L to go down, you move weights along the opposite direction of their gradients

# neuron
# input xs
# synapses with weights ws
# wi*xi + b
# bias is trigger happiness of neuron
# result is fed to activation function (usually tanh) -> squashing function
# tanh(0) = 0
# tanh(+) = capped smoothly at 1
# tanh(-) = capped smoothly at -1

# tanh = (e^2x - 1) / (e^2x + 1)

# d/dx of tanh(x) = 1 - tanh(x)^2

# topological sort allows us to call ._backward() on every node in the correct order
# we must not call ._backward() on a node until ._backward() has been called on every node that came after it

# Python uses double precision (float64) for floating point numbers by default

# in PyTorch tensors have data and grad attributes

# mean squared error loss
# loss = sum((yout - ygroundtruth)**2)
# loss.backward()

# in gradient descent we adjust parameters by small step sizes in the opposite directions of their gradients
# gradients point in direction of increasing the loss
# the step size is the learning rate

# remember to reset all the grads to zero before .backward()
# zero_grad
# this is important because grads accumulate (+=)

# binary classification example:
# training loop:

# xs = [
#   [2.0, 3.0, -1.0],
#   [3.0, -1.0, 0.5],
#   [0.5, 1.0, 1.0],
#   [1.0, 1.0, -1.0],
# ]
# 
# ys = [1.0, -1.0, -1.0, 1.0] # desired targets
# 
# n = MLP(3, [4, 4, 1])
# 
# for k in range(20):
#   
#   # forward pass
#   ypred = [n(x) for x in xs]
#   loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
#   
#   # backward pass
#   for p in n.parameters():
#     p.grad = 0.0
#   loss.backward()
#   
#   # update
#   for p in n.parameters():
#     p.data += -0.1 * p.grad
#   
#   print(k, loss.data)
