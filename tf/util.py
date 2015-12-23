
import logging

import tensorflow as tf


class VariableStore(object):

    def __init__(self, name="vs", initializer=tf.truncated_normal_initializer()):
        self.name = name
        self._default_initializer = initializer
        self.vars = {}

    def add_param(self, name, shape, initializer=None):
        initializer = initializer or self._default_initializer

        if name not in self.vars:
            with tf.name_scope(self.name + "/"):
                self.vars[name] = tf.Variable(initializer(shape), name=name)
                logging.info("Created variable %s, dim %r", self.vars[name].name,
                             shape)
        return self.vars[name]


def mlp(inp, inp_dim, outp_dim, hidden=None, f=tf.tanh, bias_output=False):
    """
    Basic multi-layer neural network implementation, with custom architecture
    and activation function.
    """

    layer_dims = [inp_dim] + (hidden or []) + [outp_dim]
    x = inp

    for i, (src_dim, tgt_dim) in enumerate(zip(layer_dims, layer_dims[1:])):
        Wi_name, bi_name = "W%i" % i, "b%i"% i

        Wi = tf.get_variable(Wi_name, (src_dim, tgt_dim))
        x = tf.matmul(x, Wi)

        is_final_layer = i == len(layer_dims) - 2
        if not is_final_layer or bias_output:
            bi = tf.get_variable(bi_name, (tgt_dim,),
                                 initializer=tf.zeros_initializer)
            x += bi

        if not is_final_layer:
            x = f(x)

    return x


def convert_labels_to_onehot(labels, batch_size, num_classes):
    """
    Convert a vector of integer class labels to a matrix of one-hot target vectors.
    """
    with tf.name_scope("onehot"):
        labels = tf.expand_dims(labels, 1)
        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        sparse_ptrs = tf.concat(1, [indices, labels], name="ptrs")
        onehots = tf.sparse_to_dense(sparse_ptrs, [batch_size, num_classes],
                                     1.0, 0.0)
        return onehots
