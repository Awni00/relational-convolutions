# Reviewer 3 (ucGQ)

Thank you for your thoughtful review. We hope to address your concerns below and look forward to further discussions with you.

## Expanding on discussion about connection to GNNs

Indeed, the connection and contrast to GNNs is an interesting facet of this work. The current 8-page version of the manuscript contains some discussion of this---with the additional page allowance, **we will expand upon this discussion in the final version**. Below is a draft of the discussion which we will add to the paper.

---

Graph neural networks can be understood through the lens of neural message-passing as follows: $h_i^{(l+1)} \gets \mathrm{Update}(h_i^{(l)}, \mathrm{Aggregate}(\{h_j^{(l)}, \ j \in \mathcal{N}(i)\}))$. That is, in a GNN, the hidden representation of node $i$ is updated as a function of the hidden representations of its neighbors on a graph. Here, the graph is part of the input to the model. This framework has proven powerful in tasks including node classification, graph classification, and link prediction. This type of model is often referred to as "relational" since the edges in the input graph can be thought of as relations.

This is in contrast to another line of work on "relational architectures," where rather than being given as input, the task-relevant relations must be inferred end-to-end [1,2,3,4,5]. Our work builds on this literature. The input in this case is merely a collection of objects, and the task requires reasoning about the relations between objects. Often, the task-relevant relations may be more complex higher-order relations. The goal of our work is to propose a natural architecture that can learn to capture such relations by composing basic building blocks.

While GNNs can be applied to such relational tasks by passing a complete graph as input along with the collection of objects, a GNN will in general lack the relational inductive biases needed to infer the task-relevant relations. This can be seen from the general form of neural message-passing. When a complete graph is given as input, each object's representation is updated as a function of *all other objects' features*. There is no step where the relations between those features are explicitly computed. In other words, the "message" being sent from $\mathcal{N}(i)$ to $i$ is an aggregation of the features of $i$'s neighbors. In a GNN, in general, the input graph is only used to direct the flow of information, but this information is about object-level features---it is not relational. In fact, a Transformer can be viewed as an instantiation of a GNN operating on a complete graph. In the case of Transformers, relations are now computed implicitly via attention scores with the effect of weighting the edges. This confers certain modeling benefits for processing relations.

By contrast, in relational convolutional networks, relations are at the center of every operation. Pairwise relations are computed *explicitly* by the MD-IPR module. Then, the resultant relation tensor is convolved against a graphlet filter to produce representations of the relational pattern in groups of objects. In particular, crucially, we emphasize that the convolution is *of the relations themselves* against the graphlet filters. This is in contrast to GNNs, where the convolution is of the objects *along* the graph (i.e., relations). This enables relational convolutional networks to *infer* the task-relevant relations end-to-end, without requiring them to be received as input like GNNs do. Moreover, our architecture is able to learn representations of iteratively more complex relational features by composing simple building blocks.


---
[1] Santoro, et al. "A simple neural network module for relational reasoning." NeurIPS 2017

[2] Shanahan, et al. "An explicitly relational neural network architecture." ICML 2020

[3] Webb, et al. "Emergent Symbols through Binding in External Memory." ICLR 2020

[4] Kerg, et al. "On Neural Architecture Inductive Biases for Relational Tasks." ICLR 2022 Workshop OSC

[5] Altabaa, et al. "Abstractors and relational cross-attention: An inductive bias for explicit relational reasoning in Transformers." ICLR 2024

---

We hope that you find this discussion interesting. We thank you again for the suggestion, we believe this addition has significantly improved our work. Does this address weakness (2) in your review? Please let us know if you have any further questions or comments.

## Complexity of computing pairwise inner products of feature maps

Thank you for the question. We discuss computational complexity of each operation of our proposed architecture at the end of Section 3.2. The complexity of computing the relation subtensors via the pairwise inner products (the step in Eq 8) is $O(n_g \cdot s^2 \cdot d_r \cdot d_{\mathrm{proj}})$. Note that these are all hyperparameters of the model and do not scale with the input size. In particular, the hyperparameter $s$ is likely to be chosen as some fixed small number such as $3$ or $5$. $n_g$ is the number of learned groups and its value should be set depending on the task. $d_r$ and $d_{\mathrm{proj}}$ are again hyperparameters (relation dimension and projection dimension). Overall, the computational complexity depends only on the scale of the model and not the scale of the input (though of course, larger models may be needed for larger more complex input).

The overall computational complexity of one layer in a relational convolutional network is also discussed in Section 3.2 and only scales linearly with the number of objects in the input. In particular, the group attention operation (Eq 7) has complexity $O(n \cdot n_g \cdot s \cdot d)$, where $n$ is the number of objects in the input, and the complexity of computing the relational convolution (Eq 9) is $O(n_g \cdot s^2 \cdot d_r \cdot n_f)$, where $n_f$ is the number of graphlet filters (also a hyperparameter). Note that *all of these operations can be computed in parallel independently* using well-optimized GPU operations.

-----

Thanks again for the well-considered review. Please let us know if we have addressed your initial concerns and if you have any further comments or questions. If we have addressed your concerns, we would be very appreciative if you could update your score accordingly.