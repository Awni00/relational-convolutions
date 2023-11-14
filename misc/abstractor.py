import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention

class RelationalAbstracter(tf.keras.layers.Layer):

    def __init__(
        self,
        num_layers,
        num_heads,
        dff,
        use_self_attn=True,
        dropout_rate=0.1,
        name=None):

        super(RelationalAbstracter, self).__init__(name=name)

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.use_self_attn = use_self_attn
        self.dropout_rate = dropout_rate

    def build(self, input_shape):

        _, self.sequence_length, self.d_model = input_shape

        # define the input-independent symbolic input vector sequence at the decoder
        normal_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.symbol_sequence = tf.Variable(
            normal_initializer(shape=(self.sequence_length, self.d_model)),
            name='symbols', trainable=True)

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.abstracter_layers = [
            RelationalAbstracterLayer(d_model=self.d_model, num_heads=self.num_heads,
                dff=self.dff, use_self_attn=self.use_self_attn,
                dropout_rate=self.dropout_rate)
            for _ in range(self.num_layers)]

        self.last_attn_scores = None

    def call(self, inputs):
        # symbol sequence is input independent, so use the same one for all computations in the given batch
        # (this broadcasts the symbol_sequence across all inputs in the batch)
        symbol_seq = tf.zeros_like(inputs)
        symbol_seq = symbol_seq + self.symbol_sequence

        symbol_seq = self.dropout(symbol_seq)

        for i in range(self.num_layers):
            symbol_seq = self.abstracter_layers[i](symbol_seq, inputs)

        return symbol_seq

class RelationalAbstracterLayer(tf.keras.layers.Layer):
  def __init__(self,
    d_model,
    num_heads,
    dff,
    use_self_attn=True,
    dropout_rate=0.1):

    super(RelationalAbstracterLayer, self).__init__()

    self.use_self_attn = use_self_attn

    if self.use_self_attn:
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

    self.relational_crossattention = RelationalAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.dff = dff
    if dff is not None:
        self.ffn = FeedForward(d_model, dff)

  def call(self, symbols, objects):
    if self.use_self_attn:
        symbols = self.self_attention(symbols)
    symbols = self.relational_crossattention(symbols=symbols, inputs=objects)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.relational_crossattention.last_attn_scores

    if self.dff is not None:
        symbols = self.ffn(symbols)  # Shape `(batch_size, seq_len, d_model)`.

    return symbols

class BaseAttention(tf.keras.layers.Layer):
    def __init__(self,
        use_residual=True,
        use_layer_norm=True,
        **kwargs):

        super().__init__()
        self.mha = MultiHeadAttention(**kwargs)
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        if use_layer_norm: self.layernorm = tf.keras.layers.LayerNormalization()
        if use_residual: self.add = tf.keras.layers.Add()


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)

        if self.use_residual:
            x = self.add([x, attn_output])
        else:
            x = attn_output

        if self.use_layer_norm:
            x = self.layernorm(x)

        return x

class RelationalAttention(BaseAttention):
  def call(self, symbols, inputs):
    attn_output, attn_scores = self.mha(
        query=inputs,
        key=inputs,
        value=symbols ,
        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    if self.use_residual:
        symbols = self.add([symbols, attn_output])
    else:
        symbols = attn_output

    if self.use_layer_norm:
        symbols = self.layernorm(symbols)

    return symbols


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
                tf.keras.layers.Dropout(dropout_rate),
            ]
        )
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x
