import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    # nin = number of inputs to the neuron
    def __init__(self, nin, nonlin=True):
        # create one weight per input
        # these are random numbers between -1 and 1
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        # bias (trigger happiness)
        self.b = Value(0)
        self.nonlin = nonlin

    # __call__ allows us to do this:
    # x = [2.0, 3.0]
    # n = Neuron(2)
    # n(x)
    def __call__(self, x):
        # zip takes 2 iterators and creates a new iterator that iterates over their corresponding entries
        # so zip pairs up the ws and the xs
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        # concatenate lists
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

# a layer of neurons is a set of neurons evaluated independently
class Layer(Module):

    # nin = number of inputs to each neuron
    # nout = number of neurons in layer
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        # get all the parameters of the neurons that compose me
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

# in a multi-layer perceptron the layers feed into each other sequentially
class MLP(Module):

    # nin = number of inputs to the MLP
    # nouts = list that defines that sizes of all the layers in our MLP
    def __init__(self, nin, nouts):
        # this is clever
        # example:
        # https://cs231n.github.io/assets/nn1/neural_net2.jpeg
        # 3 input neurons
        # 2 layers of 4
        # 1 output

        # forward pass:
        # x = [2.0, 3.0, -1.0]
        # n = MLP(3, [4, 4, 1])
        # n(x)

        # nin = 3, nouts = [4, 4, 1]
        # sz = [3, 4, 4, 1]
        # 3 inputs feed into layer of 4
        # the 4 inputs feed into layer of 4
        # the 4 inputs feed into layer of 1
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # get all the parameters of the layers that compose me
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
