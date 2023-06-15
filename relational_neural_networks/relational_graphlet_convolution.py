import itertools
import tensorflow as tf

class RelationalGraphletConvolution(tf.keras.layers.Layer):

    def __init__(
            self,
            n_filters,
            kernel_size,
            sym_inner_prod=False,
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
        kernel_size : int
            size of kernel.
        sym_inner_prod : bool, optional
            whether to use symmetric version of relational inner product, by default False.
        groups_type : str, optional
            whether groups should be permutations or combinations of size kernel_size, by default 'permutations'.
        permutation_aggregator: 'mean', 'max', or 'maxabs', optional
            how to aggregate over permutations of the groups. used when sym_inner_prod is True, by default 'mean'.
        filter_initializer : str, optional
            initializer for graphlet filters, by default 'random_normal'.
        **kwargs
            additional keyword arguments to pass to the Layer superclass (e.g.: name, trainable, etc.)
        """

        super(RelationalGraphletConvolution, self).__init__(**kwargs)

        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.sym_inner_prod = sym_inner_prod
        self.groups_type = groups_type
        self.filter_initializer = filter_initializer

        if self.sym_inner_prod:
            self.group_permutations = list(itertools.permutations(range(self.kernel_size)))
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

        self.filters_init = tf.keras.initializers.GlorotNormal()
        self.filters = self.add_weight(shape=(self.n_filters, self.kernel_size, self.kernel_size, self.rel_dim),
            initializer=self.filter_initializer, trainable=True)
        # print(f'filter.shape: {self.filters.shape}')

        if self.groups_type == 'permutations':
            self.object_groups = list(itertools.permutations(range(self.n_objects), self.kernel_size))
        elif self.groups_type == 'combinations':
            self.object_groups = list(itertools.combinations(range(self.n_objects), self.kernel_size))
        else:
            raise ValueError(f'groups_type must be permutations or combinations, not {self.groups_type}')

        # print(f'object_groups: {self.object_groups}')
        self.n_groups = len(self.object_groups)
        # print(f'n_groups: {self.n_groups}')

    @tf.function
    def rel_inner_prod(self, R_g, filters):
        if not self.sym_inner_prod:
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

    def call(self, inputs):
        # inputs: (batch_size, n_objects, n_objects, rel_dim)

        # get sub-relations
        sub_rel_tensors = tf.stack([get_sub_rel_tensor(inputs, group_indices) for group_indices in self.object_groups], axis=0)
        # sub_rel_tensors: (n_groups, batch_size, kernel_size, kernel_size, rel_dim)
        # print(f'sub_rel_tensors.shape: {sub_rel_tensors.shape}')

        # compute relational inner product
        rel_inner_prods = tf.stack([self.rel_inner_prod(sub_rel_tensors[i], self.filters) for i in range(self.n_groups)], axis=1)
        # rel_inner_prods: (batch_size, n_groups, n_filters)
        # print(f'rel_inner_prods.shape: {rel_inner_prods.shape}')

        return rel_inner_prods

    def get_config(self):
        config = super(RelationalGraphletConvolution, self).get_config()
        config.update({
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'sym_inner_prod': self.sym_inner_prod,
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
