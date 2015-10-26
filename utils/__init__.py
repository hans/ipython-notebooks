
from collections import OrderedDict
import logging

import numpy as np
import theano
from theano import tensor as T


floatX = theano.config.floatX


def init_uniform(range=0.1, dtype=floatX):
    return lambda shape: np.random.uniform(-range, range, size=shape).astype(dtype)


def init_normal(stdev=0.1, dtype=floatX):
    return lambda shape: np.random.normal(0.0, stdev, shape).astype(dtype)


def init_zeros(dtype=floatX):
    return lambda shape: np.zeros(shape, dtype=dtype)



class VariableStore(object):
    
    def __init__(self, prefix="vs", default_initializer=init_uniform()):
        self.prefix = prefix
        self.default_initializer = default_initializer
        self.vars = {}
        
    @classmethod
    def snapshot(cls, other, name=None):
        """
        Create a new `VariableStore` by taking a snapshot of another `VariableStore`.
        
        All variables in the other store will be cloned and put into a new instance.
        """
        name = name or "%s_snapshot" % other.prefix
        vs = cls(name)
        for param_name, param_var in other.vars.iteritems():
            vs.vars[param_name] = theano.shared(param_var.get_value(),
                                                borrow=False)
        return vs
        
    def add_param(self, name, shape, initializer=None):
        if initializer is None:
            initializer = self.default_initializer
            
        if name not in self.vars:
            full_name = "%s/%s" % (self.prefix, name)
            logging.debug("Created variable %s", full_name)
            self.vars[name] = theano.shared(initializer(shape),
                                            name=full_name)
            
        return self.vars[name]
    
    
def Linear(inp, inp_dim, outp_dim, vs, name="linear", bias=True):
    w = vs.add_param("%s/w" % name, (inp_dim, outp_dim))
    ret = inp.dot(w)
    
    if bias:
        b = vs.add_param("%s/b" % name, (outp_dim,), initializer=init_zeros())
        ret += b
        
    return ret


def SGD(cost, params, lr=0.01):
    grads = T.grad(cost, params)

    new_values = OrderedDict()
    for param, grad in zip(params, grads):
        new_values[param] = param - lr * grad

    return new_values


def momentum(cost, params, lr=0.01, momentum=0.9):
    grads = T.grad(cost, params)

    new_values = OrderedDict()
    for param, grad in zip(params, grads):
        param_val = param.get_value(borrow=True)
        # momentum value
        m = theano.shared(np.zeros(param_val.shape, dtype=param_val.dtype))
        # compute velocity
        v = momentum * m + lr * grad

        new_values[m] = v
        new_values[param] = param - v

    return new_values