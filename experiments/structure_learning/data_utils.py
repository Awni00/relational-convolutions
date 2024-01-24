import numpy as np
import scipy.stats


def data_generator(d, n, batch_size, dirichlet_alpha=2, n_batches=None):
    i = 0
    while n_batches is None or i < n_batches:
        xs, adjs = generate_batch(d, n, batch_size=batch_size, edge_prob=p, dirichlet_alpha=dirichlet_alpha)
        yield xs, adjs
        i += 1

def generate_batch(d, n, batch_size, edge_prob=0.5, dirichlet_alpha=2, permute=True):
    xs = []
    adjs = []
    for _ in range(batch_size):
        x, adj = generate_sample(d, n, edge_prob=edge_prob, dirichlet_alpha=dirichlet_alpha, permute=permute)
        xs.append(x)
        adjs.append(adj)

    return np.array(xs), np.array(adjs)

def generate_sample(d, n, edge_prob=0.5, dirichlet_alpha=2, permute=True):
    adj = randomly_sample_dag(d, edge_prob)

    xs = []
    for i in range(d):
        ancestors = np.where(adj[:,i] == 1)[0]
        if len(ancestors) == 0:
            xi = np.random.normal(0, 1, size=(n,))
            xs.append(xi)
            continue
        convex = scipy.stats.dirichlet(alpha=[dirichlet_alpha]*len(ancestors)).rvs()[0]
        mean = sum(convex[j] * xs[ancestors[j]] for j in range(len(ancestors)))
        xi = np.random.normal(mean, 1)
        xs.append(xi)

    xs = np.array(xs)

    if permute:
        perm = np.random.permutation(d)
        adj = adj[perm,:][:,perm]
        xs = xs[perm,:]

    return xs, adj

def randomly_sample_dag(num_nodes, edge_prob, permute=False):
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if np.random.rand() < edge_prob:
                adj[i,j] = 1

    # randomly permute
    if permute:
        perm = np.random.permutation(num_nodes)
        adj = adj[perm,:][:,perm]
        return adj, perm
    else:
        return adj