
class Value:
    """ stores a single scalar value and its gradient """

    # childen is an empty tuple by default
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        # grad stores the derivative of the output of the entire graph WRT the variable
        # that this particular node represents
        self.grad = 0

        # internal variables used for autograd graph construction

        # on init _backward is a lambda function that receives no arguments and returns None
        self._backward = lambda: None
        # on default init this is an empty set
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    # equivalent to operator+ in C++
    def __add__(self, other):
        # if other isn't of type Value, wrap it inside of Value
        other = other if isinstance(other, Value) else Value(other)

        # add the data of the two Values
        # the children of out are self and other
        out = Value(self.data + other.data, (self, other), '+')

        # note how this function is defined inside the __add__ function
        # we store it in the _backward member variable right after
        # so you can think of this function as a variable that's created here
        # this is a clever way of changing the implementation of _backward depending on
        # the operation that the node represents
        def _backward():
            # local derivatives:
            # L = s + o
            # dL/ds = 1
            # dL/do = 1
            # We multiply by out.grad because of the chain rule, which makes these derivatives global
            # We += because gradients accumulate (justification is multivariable case of the chain rule)
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    # equivalent to operator* in C++
    def __mul__(self, other):
        # if other isn't of type Value, wrap it inside of Value
        other = other if isinstance(other, Value) else Value(other)

        # multiply the data of the two Values
        # the children of out are self and other
        out = Value(self.data * other.data, (self, other), '*')

        # note how this function is defined inside the __add__ function
        # we store it in the _backward member variable right after
        # so you can think of this function as a variable that's created here
        # this is a clever way of changing the implementation of _backward depending on
        # the operation that the node represents
        def _backward():
            # local derivatives:
            # L = s * o
            # dL/ds = o
            # dL/do = s
            # We multiply by out.grad because of the chain rule, which makes these derivatives global
            # We += because gradients accumulate (justification is multivariable case of the chain rule)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"

        # ** is pow operator in Python
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            # d(x^n)/dx = n * x^(n-1)
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        # clamps all negative numbers to zero
        # is linear for all positive numbers
        #         /
        #        /
        # ______/
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    # reverse addition operator
    # when you call x + y, Python attempts to call x.__add__(y)
    # only if the method is not implemented on the left operand,
    # Python attempts to call __radd__ on the right operand and if this isn’t implemented either,
    # it raises a TypeError
    # with this, operations like 1 + Value(1) are supported
    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    # reverse subtraction operator
    def __rsub__(self, other): # other - self
        return other + (-self)

    # reverse multiplication operator
    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    # reverse division operator
    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    #def tanh(self):
    #    x = self.data
    #    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    #    out = Value(t, (self, ), 'tanh')
    #
    #    def _backward():
    #        self.grad = (1 - t**2) * out.grad
    #    out._backward = _backward
    #
    #    return out
