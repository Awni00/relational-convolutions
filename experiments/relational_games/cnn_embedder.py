import tensorflow as tf

class CNNEmbedder(tf.keras.Model):
    def __init__(self, n_f=(16,16), s_f=(3,3), pool_size=2, **kwargs):
        super(CNNEmbedder, self).__init__(**kwargs)
        self.n_f = n_f
        self.s_f = s_f
        self.pool_size = pool_size
        assert len(n_f) == len(s_f)

    def build(self, input_shape):
        self.convs = [tf.keras.layers.Conv2D(n_f, s_f, activation='relu') for n_f, s_f in zip(self.n_f, self.s_f)]
        self.pool = tf.keras.layers.MaxPool2D(self.pool_size)
        self.flatten = tf.keras.layers.Flatten()

    def encode(self, x):
        for conv in self.convs:
            x = conv(x)
            x = self.pool(x)

        x = self.flatten(x)
        return x
    
    def call(self, x):
        return tf.map_fn(self.encode, x)