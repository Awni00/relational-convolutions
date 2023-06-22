import itertools
import tensorflow as tf

class RelationalGraphletConvolution(tf.keras.layers.Layer):

    def __init__(
            self,
            n_filters,
            graphlet_size,
            symmetric_inner_prod=False,
            groups_type='permutations',
            permutation_aggregator='mean',
            filter_initializer='random_normal',
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
        groups_type : str, optional
            whether groups should be permutations or combinations of size graphlet_size, by default 'permutations'.
        permutation_aggregator: 'mean', 'max', or 'maxabs', optional
            how to aggregate over permutations of the groups. used when symmetric_inner_prod is True, by default 'mean'.
        filter_initializer : str, optional
            initializer for graphlet filters, by default 'random_normal'.
        **kwargs
            additional keyword arguments to pass to the Layer superclass (e.g.: name, trainable, etc.)
        """

        super(RelationalGraphletConvolution, self).__init__(**kwargs)

        self.n_filters = n_filters
        self.graphlet_size = graphlet_size
        self.symmetric_inner_prod = symmetric_inner_prod
        self.groups_type = groups_type
        self.filter_initializer = filter_initializer

        if self.symmetric_inner_prod:
            self.group_permutations = list(itertools.permutations(range(self.graphlet_size)))
            # print(f'group_permutations: {self.group_permutations}')

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
        # print(f'input_shape: {input_shape}')
        _, self.n_objects, _, self.rel_dim = input_shape

        self.filters = self.add_weight(shape=(self.n_filters, self.graphlet_size, self.graphlet_size, self.rel_dim),
            initializer=self.filter_initializer, trainable=True)
        # print(f'filter.shape: {self.filters.shape}')

        if self.groups_type == 'permutations':
            self.object_groups = list(itertools.permutations(range(self.n_objects), self.graphlet_size))
        elif self.groups_type == 'combinations':
            self.object_groups = list(itertools.combinations(range(self.n_objects), self.graphlet_size))
        else:
            raise ValueError(f'groups_type must be permutations or combinations, not {self.groups_type}')

        # print(f'object_groups: {self.object_groups}')
        self.n_groups = len(self.object_groups)
        # print(f'n_groups: {self.n_groups}')

    @tf.function
    def rel_inner_prod(self, R_g, filters):
        if not self.symmetric_inner_prod:
            return rel_inner_prod_filters(R_g, filters)
        else:
            permutation_rel_inner_prods = tf.stack(
                [rel_inner_prod_filters(get_sub_rel_tensor(R_g, perm), filters)
                 for perm in self.group_permutations], axis=1)
            # permutation_rel_inner_prods: (batch_size, n_permutations, n_filters)
            # print(f'permutation_rel_inner_prods.shape: {permutation_rel_inner_prods.shape}')
            agg_perm_rel_inner_prods = self.permutation_aggregator(permutation_rel_inner_prods, axis=1)
            # agg_perm_rel_inner_prods: (batch_size, n_filters)
            # print(f'agg_perm_rel_inner_prods.shape: {agg_perm_rel_inner_prods.shape}')
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
        # print(f'sub_rel_tensors.shape: {sub_rel_tensors.shape}')

        # compute relational inner product
        rel_convolution = tf.stack([self.rel_inner_prod(sub_rel_tensors[i], self.filters) for i in range(self.n_groups)], axis=1)
        # rel_convolution: (batch_size, n_groups, n_filters)
        # print(f'rel_convolution.shape: {rel_convolution.shape}')


        # if group logits are given, group the relational convolution output according to it
        if groups is not None:
            groups = tf.nn.softplus(groups) # apply softplus to ensure positive weights

            # gather weight attached to each object in group
            group_object_weights = tf.gather(tf.nn.softplus(groups), rel_graphlet_conv.object_groups, axis=-1)
            # group_weights: ([batch_size,] n_groups, graphlet_size)
            group_weights = tf.reduce_prod(group_object_weights, axis=-1)
            # group_weights: ([batch_size,] n_groups, n_graphlets)

            beta = 1 # TODO: decide whether to make this a hyperparameter/learnable parameter
            normalized_group_weights = tf.nn.softmax(beta*group_weights, axis=-1)
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
            'groups_type': self.groups_type,
            'filter_initializer': self.filter_initializer,
        })
        return config

# TODO: remove all debugging print statements (currently commented out)

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
