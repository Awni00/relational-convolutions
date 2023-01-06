"""A simple module implementing encoders to be used in relational models"""

import tensorflow as tf

class LinearProjectionEncoder(tf.keras.layers.Layer):
    """
    Linear Projection Encoder.

    Learns a 1-dimensional linear subspace of feaure space and projects along it.
    """

    def __init__(self, name=None):
        """
        create LinearProjectionEncoder layer.

        Parameters
        ----------
        name : str, optional
            name of layer, by default None
        """

        super().__init__(name=name)

    def build(self, input_shape):

        w_init = tf.random_normal_initializer() # projection vector's intializer
        self.in_dim = input_shape[-1] # dimension of features

        # define learnable projection vector
        self.proj_vec = tf.Variable(
            initial_value=w_init(shape=(self.in_dim,)),
            trainable=True, name='proj_vec')

    def call(self, inputs):

        # normalized projection vector
        norm_proj_vec, _ = tf.linalg.normalize(self.proj_vec, ord='euclidean')

        #project inputs along normalized proj_vec
        projection_length = tf.tensordot(tf.transpose(norm_proj_vec), inputs, axes=(0,1))
        projected_inputs = tf.expand_dims(projection_length, axis=-1) * norm_proj_vec

        return projected_inputs

    def get_config(self):
        return super(LinearProjectionEncoder, self).get_config()


class MLPEncoder(tf.keras.layers.Layer):
    """
    MLP Encoder Layer.

    Transforms input feature vector via a series of dense feedforward neural network layers.
    """

    def __init__(self, layer_sizes, activation='relu', name=None):
        """
        create MLP encoder layer.

        Parameters
        ----------
        layer_sizes : List[int]
            list of the number of units at each dense layer.
        activation : (list of) tensorflow activation function, optional
            tensorflow activation function or str name. if list matching
            `layer_sizes` is given, a different activation is used at each layer. by default 'relu'
        name : str, optional
            name of layer, by default None
        """

        super().__init__(name=name)

        self.layer_sizes = layer_sizes
        self.activation = activation

        # check that `activation`` is valid
        if isinstance(activation, list):
            if len(activation) != len(layer_sizes):
                raise ValueError(
                    '`activation` must either be a single activation paramter '
                    'or a list matching `layer_sizes`')
        else:
            # activation for each layer
            self.activation = [self.activation] * len(self.layer_sizes)

        # construct series of dense layers
        self.layers = []
        for layer_size, activation in zip(self.layer_sizes, self.activation):
            self.layers.append(tf.keras.layers.Dense(layer_size, activation=activation))


    def call(self, inputs):
        x = inputs

        # iteratively transform input via dense NN layers
        for layer in self.layers:
            x = layer(x)

        return x

    def get_config(self):
        config = super(MLPEncoder, self).get_config()
        config.update(
            {'layer_sizes': self.layer_sizes,
            'activation': self.activation})

        return config