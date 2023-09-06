import tensorflow as tf

class TCN(tf.keras.layers.Layer):
    """Temporal Context Normalization (Webb et al. 2021)"""

    def __init__(self, eps=1e-8, **kwargs):
        super(TCN, self).__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        _, self.n_objs, self.obj_dim = input_shape

        self.gamma = tf.Variable(tf.ones(self.obj_dim), trainable=True)
        self.beta = tf.Variable(tf.zeros(self.obj_dim), trainable=True)

    def call(self, z_seq):

        z_mu = tf.reduce_mean(z_seq, axis=1)
        z_sigma = tf.math.sqrt(tf.math.reduce_variance(z_seq, axis=1) + self.eps)
        z_seq = (z_seq - tf.expand_dims(z_mu, axis=1)) / tf.expand_dims(z_sigma, axis=1)
        z_seq = (z_seq * self.gamma) + self.beta

        return z_seq

class GroupTCN(tf.keras.layers.Layer):
    """Temporal Context Normalization performed independently across pre-specified groups"""

    def __init__(self, groups, eps=1e-8, **kwargs):
        """create GroupTCN layer.

        Parameters
        ----------
        groups : List[Tuple[int]]
            list of tuples of indices of group members.
        eps : float, optional
            small constant used for numerical stability, by default 1e-8
        """

        super(GroupTCN, self).__init__(**kwargs)
        self.eps = eps
        self.groups = groups
        self.tcn = TCN(eps=self.eps)

    def call(self, obj_seq):

        obj_seq_groups_normed = []
        for group in self.groups:
            obj_group = tf.gather(obj_seq, group, axis=1)
            obj_seq_groups_normed.append(self.tcn(obj_group))
        obj_seq = tf.concat(obj_seq_groups_normed, axis=1)

        return obj_seq