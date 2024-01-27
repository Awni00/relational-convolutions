import itertools
import tensorflow as tf
from mdipr import MultiDimInnerProdRelation
from misc.sparsemax import sparsemax

class RelationalGraphletConvolution(tf.keras.layers.Layer):

    def __init__(
            self,
            n_filters,
            graphlet_size,
            symmetric_inner_prod=False,
            groups='permutations',
            permutation_aggregator='mean',
            filter_initializer='random_normal',
            beta=1,
            group_normalizer='softmax',
            **kwargs
            ):
        """
        create RelationalGraphletConvolution layer.

        Generates a feature vector for each group of entities in a relation tensor via a graphlet convolution.

        Parameters
        ----------
        n_filters : int
            number of graphlet filters.
        graphlet_size : int
            size of graphlet.
        symmetric_inner_prod : bool, optional
            whether to use symmetric version of relational inner product, by default False.
        groups : 'permutations', 'combinations', or list of groups of size graphlet_size, optional
            whether groups should be permutations or combinations of size graphlet_size, by default 'permutations'.
        permutation_aggregator: 'mean', 'max', or 'maxabs', optional
            how to aggregate over permutations of the groups. used when symmetric_inner_prod is True, by default 'mean'.
        filter_initializer : str, optional
            initializer for graphlet filters, by default 'random_normal'.
        beta : float, optional
            temperature parameter for group logits, by default 1.
        group_normalizer : 'softmax' or 'sparsemax', optional
            whether to use softmax or sparsemax to normalize group logits, by default 'softmax'.
        **kwargs
            additional keyword arguments to pass to the Layer superclass (e.g.: name, trainable, etc.)
        """

        super(RelationalGraphletConvolution, self).__init__(**kwargs)

        self.n_filters = n_filters
        self.graphlet_size = graphlet_size
        self.symmetric_inner_prod = symmetric_inner_prod
        self.groups = groups
        self.filter_initializer = filter_initializer
        self.beta = beta
        self.group_normalizer = group_normalizer
        self.group_normalizer_ = tf.nn.softmax if group_normalizer == 'softmax' else sparsemax

        if self.symmetric_inner_prod:
            self.group_permutations = list(itertools.permutations(range(self.graphlet_size)))

            # TODO: are there any other useful aggregation functions we should consider?
            if permutation_aggregator == 'mean':
                self.permutation_aggregator = tf.math.reduce_mean
            elif permutation_aggregator == 'max':
                self.permutation_aggregator = tf.math.reduce_max
            elif permutation_aggregator == 'maxabs':
                self.permutation_aggregator = lambda x, axis: tf.math.reduce_max(tf.math.abs(x), axis=axis)
            else:
                raise ValueError(f'permutation_aggregator must be mean or max, not {permutation_aggregator}')


    def build(self, input_shape):
        _, self.n_objects, _, self.rel_dim = input_shape

        self.filters = self.add_weight(shape=(self.n_filters, self.graphlet_size, self.graphlet_size, self.rel_dim),
            initializer=self.filter_initializer, trainable=True)

        if self.groups == 'permutations':
            self.object_groups = list(itertools.permutations(range(self.n_objects), self.graphlet_size))
        elif self.groups == 'combinations':
            self.object_groups = list(itertools.combinations(range(self.n_objects), self.graphlet_size))
        elif isinstance(self.groups, list) and all(isinstance(x, tuple) and len(x) == self.graphlet_size for x in self.groups):
            self.object_groups = list(self.groups)
        else:
            raise ValueError(f"groups_type must be 'permutations' or 'combinations' or list of groups of graphlet size. recieved: {self.groups}")

        self.n_groups = len(self.object_groups)

    @tf.function
    def rel_inner_prod(self, R_g, filters):
        if not self.symmetric_inner_prod:
            return rel_inner_prod_filters(R_g, filters)
        else:
            permutation_rel_inner_prods = tf.stack(
                [rel_inner_prod_filters(get_sub_rel_tensor(R_g, perm), filters)
                 for perm in self.group_permutations], axis=1)
            # permutation_rel_inner_prods: (batch_size, n_permutations, n_filters)
            agg_perm_rel_inner_prods = self.permutation_aggregator(permutation_rel_inner_prods, axis=1)
            # agg_perm_rel_inner_prods: (batch_size, n_filters)

            return agg_perm_rel_inner_prods

    def call(self, inputs, groups=None):
        """
        computes relational convolution between inputs and graphlet filters.

        Parameters
        ----------
        inputs : tf.Tensor
            relation tensor of shape (batch_size, n_objects, n_objects, rel_dim).
        groups: tf.Tensor, optional
            tensor assigning weights to each object in each group,
            of shape (n_groups, n_objects) or (batch_size, n_groups, n_objects).

        Returns
        -------
        tf.Tensor
            result of relational convolution of shape (batch_size, n_groups, n_filters)
        """

        # get sub-relations
        sub_rel_tensors = tf.stack([get_sub_rel_tensor(inputs, group_indices) for group_indices in self.object_groups], axis=0)
        # sub_rel_tensors: (n_groups, batch_size, graphlet_size, graphlet_size, rel_dim)

        # compute relational inner product
        rel_convolution = tf.stack([self.rel_inner_prod(sub_rel_tensors[i], self.filters) for i in range(self.n_groups)], axis=1)
        # rel_convolution: (batch_size, n_groups, n_filters)

        # if group logits are given, group the relational convolution output according to it
        if groups is not None:
            groups = tf.nn.softplus(groups) # apply softplus to ensure positive weights

            # gather weight attached to each object in group
            group_object_weights = tf.gather(groups, self.object_groups, axis=-1)
            # group_weights: ([batch_size,] n_groups, graphlet_size)
            group_weights = tf.reduce_prod(group_object_weights, axis=-1)
            # group_weights: ([batch_size,] n_groups, n_graphlets)

            normalized_group_weights = self.group_normalizer_(self.beta*group_weights, axis=-1)
            # normalized_group_weights: ([batch_size,] n_groups, n_graphlets)

            if len(tf.shape(groups)) == 2:
                # y_bgr = sum_n alpha_gn x_bnr
                rel_convolution = tf.einsum('bnr,gn->bgr', rel_convolution, normalized_group_weights)
            elif len(tf.shape(groups)) == 3:
                # y_bgr = sum_n alpha_bgn x_bnr
                rel_convolution = tf.einsum('bnr,bgn->bgr', rel_convolution, normalized_group_weights)
            else:
                raise ValueError(f'groups must have shape (n_groups, n_objects) or (batch_size, n_groups, n_objects), not {tf.shape(groups)}')
            # rel_convolutions: (batch_size, n_groups, n_filters)

        return rel_convolution

    def get_config(self):
        config = super(RelationalGraphletConvolution, self).get_config()
        config.update({
            'n_filters': self.n_filters,
            'graphlet_size': self.graphlet_size,
            'symmetric_inner_prod': self.symmetric_inner_prod,
            'groups_type': self.groups,
            'filter_initializer': self.filter_initializer,
        })
        return config


class RelationalGraphletConvolutionGroupAttn(tf.keras.layers.Layer):

    def __init__(
            self,
            n_filters,
            graphlet_size,
            n_groups,
            rel_dim,
            group_attn_key_dim,
            group_attn_key='pos+feat',
            symmetric_inner_prod=False,
            permutation_aggregator='mean',
            filter_initializer='random_normal',
            beta=None,
            group_normalizer='softmax',
            **kwargs
            ):
        """
        create RelationalGraphletConvolutionGroupAttn layer.

        Groups objects via attention, computes relation tensor within each group, and computes relational convolution. 
        Generates a feature vector for each group of entities via a graphlet convolution.

        Parameters
        ----------
        n_filters : int
            number of graphlet filters.
        graphlet_size : int
            size of graphlet.
        n_groups : int
            number of learned groups, by default 'permutations'.
        rel_dim : int
            dimension of relation tensor.
        group_attn_key_dim : int
            dimension of query/key transformations for group attention.
        group_attn_key : 'pos', 'feat', or 'pos+feat', optional
            whether to use positional embeddings, feature embeddings, or both for group attention, by default 'pos+feat'.
        symmetric_mdipr : bool, optional
            whether to use symmetric version of MDIPR, by default False.
        symmetric_inner_prod : bool, optional
            whether to use symmetric version of relational inner product (i.e., aggregate across permutations), by default False.
        permutation_aggregator: 'mean', 'max', or 'maxabs', optional
            how to aggregate over permutations of the groups. used when symmetric_inner_prod is True, by default 'mean'.
        filter_initializer : str, optional
            initializer for graphlet filters, by default 'random_normal'.
        beta : float, optional
            temperature parameter for group logits, by default None.
        group_normalizer : 'softmax' or 'sparsemax', optional
            whether to use softmax or sparsemax to normalize group logits, by default 'softmax'.
        **kwargs
            additional keyword arguments to pass to the Layer superclass (e.g.: name, trainable, etc.)
        """

        super(RelationalGraphletConvolutionGroupAttn, self).__init__(**kwargs)

        self.n_filters = n_filters
        self.graphlet_size = graphlet_size
        self.n_groups = n_groups
        self.rel_dim = rel_dim
        self.group_attn_key_dim = group_attn_key_dim
        self.group_attn_key = group_attn_key # TODO: add support for 'contextual' key (i.e., perform self-attention first)
        self.symmetric_inner_prod = symmetric_inner_prod
        self.filter_initializer = filter_initializer
        self.beta = beta
        self.group_normalizer = group_normalizer
        if group_normalizer == 'softmax':
            self.group_normalizer_ = tf.nn.softmax
        elif group_normalizer == 'sparsemax':
            self.group_normalizer_ = sparsemax
        else:
            raise ValueError(f'group_normalizer must be softmax or sparsemax, not {group_normalizer}')

        if self.symmetric_inner_prod:
            self.group_permutations = list(itertools.permutations(range(self.graphlet_size)))

            # TODO: are there any other useful aggregation functions we should consider?
            if permutation_aggregator == 'mean':
                self.permutation_aggregator = tf.math.reduce_mean
            elif permutation_aggregator == 'max':
                self.permutation_aggregator = tf.math.reduce_max
            elif permutation_aggregator == 'maxabs':
                self.permutation_aggregator = lambda x, axis: tf.math.reduce_max(tf.math.abs(x), axis=axis)
            else:
                raise ValueError(f'permutation_aggregator must be mean or max, not {permutation_aggregator}')


    def build(self, input_shape):
        _, self.n_objects, self.obj_dim = input_shape

        self.filters = self.add_weight(shape=(self.n_filters, self.graphlet_size, self.graphlet_size, self.rel_dim),
            initializer=self.filter_initializer, trainable=True)

        self.mdipr = MultiDimInnerProdRelation(rel_dim=self.rel_dim, proj_dim=self.rel_dim, symmetric=True)

        self.group_queries = tf.keras.layers.Embedding(input_dim=self.n_groups*self.graphlet_size, output_dim=self.group_attn_key_dim)
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=self.n_objects, output_dim=self.group_attn_key_dim)
        self.group_attn_key_map = tf.keras.layers.Dense(self.group_attn_key_dim, use_bias=False)

        if self.beta is None:
            self.beta = self.group_attn_key_dim**(-0.5)

    def group_attention(self, inputs):

        batch_dim, seq_len, obj_dim = tf.shape(inputs)
        q = self.group_queries(tf.range(self.n_groups*self.graphlet_size))
        if self.group_attn_key == 'pos':
            k = tf.repeat(tf.expand_dims(self.pos_embedding(tf.range(seq_len)), axis=0), batch_dim, axis=0)
        elif self.group_attn_key == 'feat':
            k = self.group_attn_key_map(inputs)
        elif self.group_attn_key == 'pos+feat':
            k = self.group_attn_key_map(inputs) + self.pos_embedding(tf.range(seq_len))
        else:
            raise ValueError(f'group_attn_key must be pos, feat, or pos+feat, not {self.group_attn_key}')
        v = inputs

        attn_scores = self.group_normalizer_(self.beta * tf.matmul(q, k, transpose_b=True)) # (n_groups*filter_size, seq_len)
        attn_output = tf.matmul(attn_scores, v) # (n_groups*filter_size, obj_dim)

        attn_scores = tf.reshape(attn_scores, (-1, self.n_groups, self.graphlet_size, seq_len)) # (batch_dim, n_groups, filter_size, seq_len)
        attn_output = tf.reshape(attn_output, (-1, self.n_groups, self.graphlet_size, obj_dim)) # (batch_dim, n_groups, filter_size, obj_dim)

        return attn_output, attn_scores


    @tf.function
    def rel_inner_prod(self, R_g, filters):
        if not self.symmetric_inner_prod:
            return rel_inner_prod_filters(R_g, filters)
        else:
            permutation_rel_inner_prods = tf.stack(
                [rel_inner_prod_filters(get_sub_rel_tensor(R_g, perm), filters)
                 for perm in self.group_permutations], axis=1)
            # permutation_rel_inner_prods: (batch_size, n_permutations, n_filters)
            agg_perm_rel_inner_prods = self.permutation_aggregator(permutation_rel_inner_prods, axis=1)
            # agg_perm_rel_inner_prods: (batch_size, n_filters)

            return agg_perm_rel_inner_prods

    def call(self, inputs):
        """
        computes relational convolution between inputs and graphlet filters.

        Parameters
        ----------
        inputs : tf.Tensor
            relation tensor of shape (batch_size, n_objects, n_objects, rel_dim).

        Returns
        -------
        tf.Tensor
            result of relational convolution of shape (batch_size, n_groups, n_filters)
        """

        # get object groups via group attention
        obj_groups, group_attn_scores = self.group_attention(inputs)

        # get sub-relations
        sub_rel_tensors = tf.stack([self.mdipr(attn_output[:, group]) for group in range(n_groups)], axis=1)
        # sub_rel_tensors: (batch_size, n_groups, graphlet_size, graphlet_size, rel_dim)

        # compute relational inner product
        rel_convolution = tf.stack([self.rel_inner_prod(sub_rel_tensors[:, i], self.filters) for i in range(self.n_groups)], axis=1)
        # rel_convolution: (batch_size, n_groups, n_filters)

        return rel_convolution

    def get_config(self):
        config = super(RelationalGraphletConvolutionGroupAttn, self).get_config()
        config.update({
            'n_filters': self.n_filters,
            'graphlet_size': self.graphlet_size,
            'n_groups': self.n_groups,
            'group_attn_key_dim': self.group_attn_key_dim,
            'group_attn_key': self.group_attn_key,
            'symmetric_inner_prod': self.symmetric_inner_prod,
            'filter_initializer': self.filter_initializer,
            'beta': self.beta,
            'group_normalizer': self.group_normalizer
        })
        return config



def get_sub_rel_tensor(R, group_indices):
    """
    get sub-tensor of R composed of relations within the given group's indices.

    Parameters
    ----------
    R : tf.Tensor
        relation tensor of shape (batch_size, n_objects, n_objects, rel_dim)
    group_indices : Tensor[int], List[int], or Tuple[int]
        indices of group members of shape (group_size, )

    Returns
    -------
    tf.Tensor
        sub_rel_tensor of shape (batch_size, group_size, group_size, rel_dim)
    """

    # get rows then columns of R corresponding to group members
    sub_rel_tensor = tf.gather(tf.gather(R, group_indices, axis=1), group_indices, axis=2)

    return sub_rel_tensor

def rel_inner_prod_filter(R_g, filter_):
    """
    compute relational inner product between relation tensor and filter_.

    Parameters
    ----------
    R_g : tf.Tensor
        relation tensor of shape (batch_size, group_size, group_size, rel_dim)
    filter_ : tf.Tensor
        graphlet filter of shape (group_size, group_size, rel_dim)

    Returns
    -------
    tf.Tensor
        relational inner product of shape (batch_size, )
    """

    return tf.reduce_sum(R_g * filter_, axis=[1,2,3])

def rel_inner_prod_filters(R_g, filters):
    """
    compute relational inner product between relation tensor and stack of filters.

    Parameters
    ----------
    R_g : tf.Tensor
        relation tensor of shape (batch_size, group_size, group_size, rel_dim)
    filters : tf.Tensor
        stack of graphlet filters of shape (n_filters, group_size, group_size, rel_dim)

    Returns
    -------
    tf.Tensor:
        result of relational inner product of shape (batch_size, n_filters)
    """

    return tf.stack([rel_inner_prod_filter(R_g, filters[i]) for i in range(filters.shape[0])], axis=1)
