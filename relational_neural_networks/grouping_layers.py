import tensorflow as tf
import numpy as np

class FeatureGrouping(tf.keras.layers.Layer):
    def __init__(self, num_groups, mlp_shape, mlp_activations='relu', use_pos=True, **kwargs):
        super().__init__(**kwargs)

        self.num_groups = num_groups
        self.mlp_shape = mlp_shape
        self.mlp_activations = mlp_activations
        if isinstance(mlp_activations, list) and len(mlp_activations) != len(mlp_shape):
            raise ValueError(f'mlp_activations must be the same shape as mlp_shape.')
        else:
            self.mlp_activations = [mlp_activations]*len(mlp_shape)
        self.use_pos = True

    def build(self, input_shape):

        batch_size, num_objects, object_dim = input_shape

        dense_layers = [
            tf.keras.layers.Dense(n_units, activation=activation)
            for n_units, activation in zip(self.mlp_shape, self.mlp_activations)
        ]
        dense_layers.append(tf.keras.layers.Dense(self.num_groups, activation='linear'))

        self.feature_grouping_mlp = tf.keras.Sequential(dense_layers,
            name='feature_grouping_mlp')

        if self.use_pos:
            self.add_positional_embedding = AddPositionalEmbedding(max_length=num_objects)

    def call(self, inputs):
        # inputs: sequence of objects of shape (batch_size, n_objects, obj_dim)
        # output: group matrix of shape (batch_size, n_objects, num_groups)

        if self.use_pos:
            inputs = self.add_positional_embedding(inputs)

        group_matrix = self.feature_grouping_mlp(inputs)
        group_matrix = tf.transpose(group_matrix, perm=(0, 2, 1))
        return group_matrix

class TemporalGrouping(tf.keras.layers.Layer):
    def __init__(self, num_groups, weight_initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.num_groups = num_groups
        self.weight_initializer = weight_initializer

    def build(self, input_shape):

        _, n_objects, obj_dims = input_shape

        self.w_init = tf.keras.initializers.get(self.weight_initializer)

        # define learnable group logits
        self.group_matrix = tf.Variable(
                initial_value=self.w_init(shape=(self.num_groups, n_objects)),
                trainable=True,
                name=f'group_matrix')

    def call(self, inputs):
        return self.group_matrix


# region utils
def create_positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class AddPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_length=1024, name="add_positional_embedding"):
        super().__init__(name=name)
        self.max_length = max_length

    def build(self, input_shape):
        _, self.seq_length, self.vec_dim = input_shape
        self.max_length = max(self.max_length, self.seq_length)
        self.pos_encoding = create_positional_encoding(length=self.max_length, depth=self.vec_dim)

    def call(self, x):
        length = tf.shape(x)[1]

        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.vec_dim, tf.float32))

        # add positional encoding
        x = x + self.pos_encoding[tf.newaxis, :length, :]

        return x
# endregion