Thank you for your review and your helpful thoughts and feedback.

We have uploaded a revised version of the manuscript with several additions to address feedback from reviews. Please see the global response for an overview of these additions. Below, we will address your particular concerns and highlight additions related to your review in more detail.

> $\langle x_1 W_1, x_2 W_2\rangle = x_2 W_2 W_1 x_1^\top = x_2 W x_1$ has interesting properties when $W$ is positive definite, and also it’s interesting that if $d_{\mathrm{proj}} < d_{\phi}$ then $W_2 W_1^\intercal$ is a low-rank approximation of $W$. I wonder if the authors have thought about this.

Yes. Both of these points are important considerations. There are two parts to your question that we will respond to in turn. We refer to section 2 in the paper and provide a discussion below.

***Symmetric relations and positive (semi)-definiteness of $W$.***
As you point out, when $W \in \reals^{d \times d}$ is symmetric positive (semi-)definite, the relation $r(x_1, x_2) = x_1^\intercal W x_2$ has some additional structure. In the paper, we discuss this in terms of the *symmetry* of the relation $r(x, y)$. In our model, $W$ is parameterized by $W_1, W_2 \in \mathbb{R}^{d_{\mathrm{proj}} \times d}$ as $W:= W_1^{\intercal} W_2$. In the symmetric case, $W_1 = W_2$ and $W$ is symmetric positive semi-definite. This, of course, is a common way to parameterize PSD matrices (see e.g., ”Cholesky decomposition”). Symmetry is an inductive bias that is useful in some tasks (but may be detrimental in others). In particular, as we mention in section 2, symmetric relations obey a transitivity property since $r(x, y)$ induces a pseudometric.

We think of $W_1, W_2$ as feature filters that the inner product relation compares. When $W_1 = W_2$, the feature extracted from $x_1$ is the same as the feature extracted from $x_2$, and hence the relation $r(x_1, x_2)$ is comparing the same attribute across the two objects.  When $W_1 \neq W_2$, the relation $r(x_1, x_2)$ is asymmetric in the sense that it is comparing the two objects along different dimensions. For example, in the symmetric case, $r(x_1, x_2)$ may represent a relation like “the color is the same”, whereas in the asymmetric case, $r(x_1, x_2)$ may represent a relation like “$x_1$  is bigger than $x_2$”.

Intuitively, symmetry is a useful inductive for the tasks we consider in our experiments since the task-relevant relations are symmetric similarity relations (e.g., in *SET!*, the relevant relations are same/different across shape, color, number, and fill). The variant of RelConvNet in our experiments uses symmetric relations with $W_1 = W_2$. We have added further discussion to the paper on the effect of symmetry as an inductive bias for the tasks we consider. (More on this below.)

***Low-rank $W$.***
The matrix $W$ will be at most rank $d_{\mathrm{proj}}$ when $d_{\mathrm{proj}} < d_{\phi}$ (which is always the case in our experiments). The rank of $W$ should be thought of as the dimensionality of the feature being filtered and compared. Typically, one would like to perform a comparison along a particular dimension or “feature”. Recall that $W_1, W_2$ are projecting features to be compared from the original full feature representation. A “feature” is captured by a $d_{\mathrm{proj}}$-subspace of the original $d_\phi$-dimensional feature representation.

We are unsure what you meant by $W_1^\intercal W_2$ being an “approximation” of something; perhaps you just meant to say that it is a parameterization of a low-rank matrix. But please let us know if there is a part of your question that we missed.

> The text mentions that pairwise computations are a limitation, but really the number of possible groups is $2^n$ (or if we limit ourselves to size 𝑠 groups); isn't that the "scary number"? Transformers work "just fine" in $O(n^2)$ (and reasonable $O(n \log n)$ versions exist).

Recall that in the case of discrete groupings (Section 3.1, Eq 5), the relational convolution takes the form
$$R \ast \boldsymbol{f} := \left(\langle R[g], \boldsymbol{f} \rangle_{\mathrm{rel}}\right)_{g \in \mathcal{G}},$$
where $\mathcal{G}$ is the set of discrete groups. The computational cost of this operation is $|\mathcal{G}| \cdot s^2 \cdot d_r \cdot n_f$, where $s$ is the graphlet filter size, $d_r$ is the relation dimension, and $n_f$ is the number of filters. If $\mathcal{G}$ is, say, all combinations of size $s$, $|\mathcal{G}| = \binom{n}{s} \asymp n^s$. Typically, $s > 2$ (to be interesting, otherwise it is still representing a two-way relation), so this is worse than the $O(n^2)$ computational complexity of Transformers.

*The bigger issue is that $|\mathcal{G}|$ becomes the number of objects at the next layer.* That is, the input to the next layer is a sequence of $|\mathcal{G}|$ objects, with the next layer then needing to compute a $|\mathcal{G}| \times |\mathcal{G}|$ relation tensor, etc. This causes a combinatorial explosion in the number of objects at deeper layers (e.g., number of objects at layer $\ell$ is $n_\ell = \binom{n_{\ell-1}}{s}$).

Explicitly modeling the task-relevant groups as we propose enables us to keep the number of objects at each layer a fixed amount, as a controllable hyperparameter.

> About the attention grouping, the text notes: "learn $n_g$ groupings of objects, retrieving $s$ objects per group." I'm a bit ambivalent w.r.t. describing this as "retrieving objects". In the limit, the attention operation in (6) can just average all the objects' embeddings, did it really "retrieve" something or did it do some intermediate aggregation operation?

Your point about terminology is well-taken. Of course, attention is not literally performing a discrete retrieval operation. It is rather a differentiable soft-retrieval operation. The softmax in attention is a differentiable version of an argmax, and in the limit (e.g., as temperature $1/\beta \to 0$) this recovers a hard-assignment. We use the term "retrieval" as an analogy to explain what we think of the attention step as implementing. We note also that the entropy regularization promotes learning an attention operation that is closer to a discrete assignment (see below).

> How does entropy change during training? Are different factors more or less effective?

We've added an exploration of the effect of entropy regularization, at different levels of regularization, to the appendix. We focus our exploration on the "match pattern" task. We will summarize the results here. In Figure 9 of the updated manuscript, we explore the trade-offs between different levels of regularization on the task loss and group attention entropy at the end of training. We find that entropy regularization is necessary for learning the task. Without entropy regularization, the task loss does not decrease. But with even a small amount of regularization, the task loss is able to escape the local minimum at initialization. As the level of regularization decreases, the group attention distribution's entropy at convergence decreases, converging to discrete assignments. In Figure 10, we plot the evolution of the task loss and entropy regularization loss over the course of training. The entropy regularization term starts at $\log(n)$ (the entropy of a uniform distribution) and decreases monotonically while the base cross-entropy loss also decreases monotonically. Expectedly, the entropy of the group attention scores at convergence is smaller when the regularization scaling factor is larger. At initialization, group attention is essentially computing an average with uniform weights, and the relations between these averaged objects contain too little information for the task loss to guide the model in a useful direction. However, with a little bit of entropy regularization, the model is able to escape the local minimum at which point the task loss is able to guide the model towards a solution.

> Regularizing attention entropy for sparsity is something that's been done before, please reference prior work.

We've added some references.

TODO: add to paper

> One elephant in the room is what happens when objects are not neatly separated in the input space? or when the notion of what's an object or a collection of objects is blurry?

This is an interesting and important question. Indeed, the modules we developed in this paper assume object-centric representations as input (e.g., a sequence of vector embeddings each corresponding to an object in the scene). The task of learning such object-centric representations from raw perceptual inputs is an active area of research. It is sometimes referred to as *object-discovery*. The problem of learning object-centric representations is related but separate to the problem of learning relational representations. In our paper, we tackle the latter.

Learning object-centric representations can be done in an unsupervised way. For example, one notable piece of work in this area is Locatello et al.'s *Slot Attention*. The output of such methods is a sequence of vector embeddings, each describing an object in the scene. The output of something like Slot Attention can produce the input to relational convolutional networks to yield an end-to-end system learning from raw perceptual inputs. An important direction of future work will be to explore how well relational convolutional networks integrate with object-discovery methods like Slot Attention, and whether such a system could successfully learn to perform abstract relational reasoning in complex scenes.

We will add a brief discussion on the connection between learning object-centric representations and learning relational representations.

TODO: add to paper

> I wonder if a regular deep ConvNet baseline would make sense? ... The "Common CNN Embedder" described in Table 2 feels incredibly shallow.

First, a point of clarification. The "Common CNN Embedder" is a CNN module which is applied independently to each of the 9 objects in the $3 \times 3$ grid to produce a sequence of 9 embedding vectors. Each object (patch of image) is quite small; only $12 \times 12 \times 3$. So, a shallow CNN embedder is reasonable. The CNN only needs to extract very low-level features like color and shape.

Nonetheless, your question stands as to whether a regular CNN could learn to perform these tasks end-to-end from the raw image input. We have carried out additional experiments to explore this. Interestingly, CNNs were not included as a baseline in the experiments of related work on relational architecture. There seems to be an ambient assumption in the literature on relational architectures that a CNN would be unable to learn such tasks since it lacks explicit relational processing capabilities. Given your question, we were curious to confirm whether this assumption was correct.

We tested a 8-layer CNN model on the relational games tasks and a 10-layer model on the *Set* task, where the input to these models is now the raw image input. The results were interesting. On some of the relational games tasks (the simpler ones) the CNN model was perfectly capable of learning and generalizing. In fact, the CNN was on-par with the top-performing models on the "same" and "occurs" tasks. However, on the more difficult "xoccurs" and "match pattern" tasks, the CNN was the worst-performing model, with accuracy stuck at 50\%. Similarly, on the *Set* task, the CNN model was completely unable to learn the task in a manner that generalizes and, unlike certain other models like the Transformer, it was not even able to fit the training data.

Overall, we observe in our experiments that certain simpler tasks are solvable by a broader range of architectures, giving little separation between different architectures, while more difficult tasks give a drastic separation (oftentimes, models completely fail to learn in a manner that generalizes). This highlights the need for additional relational reasoning benchmarks that enable further evaluation of relational architectures.

A discussion on these additional results has been added to the paper.

> Final point, I know it may be unfair to compare to highly optimized transformer CUDA kernels, but I couldn't find it mentioned how the method fares computationally in practice. Does it seem to scale well?

At the end of Section 3.2, we discuss computational efficiency. We note that the overall computational complexity of a relational convolution layer (with group attention) is $O(n \cdot n_g \cdot s \cdot d + n_g \cdot s^2 \cdot d_r \cdot \max(d_{\mathrm{proj}}, n_f))$, where $n$ is the number of input objects, and the rest are hyperparameters. This can be implemented efficiently in modern deep learning libraries like PyTorch/Tensorflow/Jax/etc. In particular, computing the relation tensors can be computed in parallel with efficient matrix-matrix multiplications, and the group attention operation can use modern fast kernels like FlashAttention. In our experiments, the run-time of the RelConvNet experimental runs was similar to the Transformer baseline. Detailed experimental logs (including run-time, resource usage, metrics, etc.) are available through an online portal linked through the open-source code, which will be included in the de-anonymized version of the paper.
