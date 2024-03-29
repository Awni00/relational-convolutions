# Reviewer 4 (wwyA)

Thank you for your review. We appreciate your positive feedback on the clarity of our writing and the strength of our experimental results.

The review raises two weaknesses. However, both are based on a crucial misunderstanding of our proposed architecture. Thus, the two stated weaknesses are factually incorrect. In this rebuttal, we aim to explain and clarify, point by point.


**Weakness 1**

In weakness 1 of your review, you describe our proposal as:
>pairwise relation for the input is the "attention matrix", the paper then uses convolution network on top of the attention matrix.

You cite this as a limitation to the novelty of our work. *This is **not** the architecture we propose*. Our proposed method does not use spatial 2D convolutions. Rather, we propose a *novel* convolution-like operation which we call "relational convolutions". This is distinct from the spatial convolutions of CNNs, despite the similarity in name, and addresses several key challenges in the domain of relational representation. Relational convolutions are similar only *in spirit* to spatial convolutions (hence the name), but the mathematical formulation is entirely different.

**Weakness 2**

In addition, the statement about the computational complexity of our method in weakness 2 is also incorrect (it seems as a result of this misunderstanding):
>I assume the total complexity is at least O(n^3) (n is input number), cannot scale.

We would like to clarify that *the computational complexity scales ***linearly*** with the number of inputs*, as explained in Section 3.2 (L215-242) of the paper. In fact, achieving this favorable computational complexity requires solving key technical challenges that are addressed by our proposed architecture.

We will explain both of these points in detail below. We aim to clarify any confusion and provide an overview of our proposed method, highlighting its novelty and distinguishing features.

## Understanding the relational convolutions architecture: correcting some misunderstandings (weakness 1)

> Novelty is limited and some part is confusion. pairwise relation for the input is the "attention matrix", the paper then uses convolution network on top of the attention matrix. why choose convolution on this level of reasoning?

**Our proposed architecture does *not* involve applying a convolutional network on top of the attention matrix**. As you mention, the inductive bias for CNNs is spatial locality, which does not make sense as a way to process the relation tensor produced by the MD-IPR layer. **The "relational convolution" operation is a novel operation we propose that is distinct from the spatial convolution operation in CNNs**. In particular, **in relational convolutions, the convolution is with respect to *groupings of objects* (which can be learned), not spatial translations as in the spatial convolutions of CNNs.**

We will explain the proposed architecture below in more detail and emphasize how it differs from the spatial convolutions of CNNs.

### High-level overview

First, we summarize the architecture at a high level.

1. Pairwise relations between object features are computed using the multi-dimensional inner product relations (MD-IPR) module. This is achieved by learning feature maps that extract or 'filter' attributes for comparison through an inner product.
2. A set of **graphlet filters** are learned which represent a template of the relational pattern within a group of objects. The graphlet filters are "convolved" with the relation tensor through a novel **relational convolution** operation. This operation computes a *representation of the relational pattern in each group of objects*.
3. We propose an attention-based operation, which we call **"group attention"**, to *learn the relevant groupings end-to-end*. Group attention retrieves the relevant objects for each group *differentiably*. This approach alleviates the need to consider a combinatorial number of groups during the computation of the relational convolution, addressing a key technical challenge (this is part of how we achieve linear complexity).
4. The **relational convolution** operation returns a sequence of new "objects" (i.e., a sequence of vectors), with each object now representing the relational pattern within a group of input objects. This enables the overall operation to be repeated in order to *learn representations of higher-order relations*.

With this high-level overview, we now describe the relational convolution operation in more technical detail. We will draw the contrast to standard spatial convolutions for both relational convolutions with discrete groups (Section 3.1 of the paper) and relational convolutions with group attention (Section 3.2 of the paper).

### Relational Convolutions with Discrete Groups

In this case, we consider a setting where a collection of discrete groups $\mathcal{G}$ is provided as input to the model. The first step is to compute the $n \times n \times d_r$ relation tensor using the MD-IPR layer.

We learn $n_f$ different graphlet filters, each of size $s \times s \times d_r$, where $s$ is the size of the graphlet (number of objects in each group). These act as templates of the relational pattern in a group of objects. $n_f$ and $s$ are both hyperparameters.

*The relational convolution operation compares the pairwise relations within groups of $s$ objects against the graphlet filters*. For a group $g \subset [n]$ of $s$ objects, the pairwise relations within this group are given by the relation subtensor $R[g] := [R[i,j]]_{i,j \in g}$. The relation subtensor is compared against the graphlet filters $\boldsymbol{f} = (f_1, \ldots, f_{n_f}) \in \mathbb{R}^{n_f \times s \times s \times d_r}$ to produce an $n_f$-dimensional vector summarizing the relational pattern in group $g$ (see Eqs 3,4 in main text).

The graphlet filters are convolved *across groups* $g \in \mathcal{G}$ to produce a sequence of $|\mathcal{G}|$ objects as follows,
$$R * \boldsymbol{f} := (\langle R[g], \boldsymbol{f}\rangle_{\mathrm{rel}})_{g \in \mathcal{G}}$$

*In a relational convolution operation, the convolution is performed across **groups**, rather than spatial translations as in the 2D convolutions of CNNs.* In particular, $\boldsymbol{f}$ is matched against the groups in $\mathcal{G}$, whereas in a spatial convolution, filters would be matched against *contiguous* patches of the relation tensor (e.g., $[0:s], [1:s+1], [2:s+2], ..., [n-s:n]$, etc.). As you point out, this would not make sense in the context of relational processing. *To emphasize this contrast to the reader and highlight the novelty of our approach, we will add a brief discussion to the paper explaining the difference between relational convolutions and spatial convolutions.*

### Relational Convolutions with Group Attention

In some situations, the set of task-relevant discrete groups may not be available and considering all possible groups may be impractical. To address this setting, we propose a variant of relational convolutions in which groups are learned and represented differentiably via an attention operation. This is a key technical challenge addressed by our work.

Below, we will summarize the methodology. Please see section 3.2 of the paper for the complete technical description.

1. First, for each of $n_g$ groups (with $n_g$ being a hyperparameter) we retrieve $s$ objects from the collection of $n$ objects. We denote these objects by $\bar{x}_i^g$, where $g \in [n_g], i \in [s]$. (See Eq 7)
2. Next, we compute the relation tensor for each group $g \in [n_g]$ using a shared MD-IPR layer producing $\bar{R}[g] \in \mathbb{R}^{s \times s \times d_r}$. That is, the MD-IPR layer computes the relations within the *retrieved* objects in each group. (See Eq 8)
3. For each group $g$, $\bar{R}[g]$ is compared against the graphlet filters $\boldsymbol{f}$ producing the relational convolution $\bar{R} * \boldsymbol{f}$, which consists of a sequence of $n_g$ vectors, each summarizing the relational pattern within a learned group. (See Eq 9)

As you can see, this is *entirely distinct* from spatial convolutions and introduces several architectural innovations enabling a powerful (and computationally-efficient) way to learn complex relational representations.

---

Does this clarify your confusion about the proposed architecture and the contrast to spatial convolutions? Please let us know if you have further questions. We look forward to your response.

<!-- should we omit this section? it's a less important distinction and may be lost on the reviewer... -->
<!-- ### MD-IPR vs Attention Scores -->

<!-- Finally, a minor comment about multi-dimensional inner product relations vs attention scores. Indeed, the MD-IPR layer bears a resemblance to the attention matrix of Transformers. A couple of clarifications are in order. This is not an "attention" matrix because it is not used in a "retrieval" operation (i.e., in a Transformer, attention computes a convex combination of objects in the sequence weighted by the attention scores). That is, in Transformers, the attention matrix is an intermediate step used to direct the flow of object-level features. In MD-IPR, the relation tensor is the *final output* of the MD-IPR layer and is used directly as a representation of the relations. Related to this, the relation tensor is not normalized via a softmax as in attention. -->

## Computational Efficiency & Scalability (weakness 2)
> The algorithm uses convolution network on top of the attention matrix, which is very slow. I assume the total complexity is at least O(n^3) (n is input number), cannot scale.

As explained above, our architecture is not to "use [a] convolutional network on top of the attention matrix" and hence **the conclusion about computational complexity and scalability in weakness 2 of the review is factually incorrect**. The **computational complexity is in fact linear in $n$**, the number of objects in the input. **Section 3.2 (L215-242) in the paper discusses the computational complexity of each step in the relational convolution layer**.

To make the rebuttal self-contained, we will summarize the discussion on computational complexity in the paper. The group attention operation (Eq 7) has complexity $O(n \cdot n_g \cdot s \cdot d)$, where $n$ is the number of objects in the input. Computing the relation tensors for each group $\bar{R}[g]$ (Eq 8) has complexity $O(n_g \cdot s^2 \cdot d_r \cdot d_{\mathrm{proj}})$. Finally, the complexity of computing the relational convolution (Eq 9) is $O(n_g \cdot s^2 \cdot d_r \cdot n_f)$, where $n_f$ is the number of graphlet filters (also a hyperparameter). Note that *all of these operations can be computed in parallel independently* using well-optimized GPU operations. Moreover, the first operation (group attention) scales only linearly with input size, and the latter two scale only with the model size (i.e., hyperparameters) and not the size of the input.


----
<!-- reviewer does not ask about this. should we include it? not sure what kind of person we're dealing with here. if a long rebuttal would dissuade them from reading, we should probably not include this... it may also mask emphasis on the factual errors in the review. -->
## Contrast to GNNs

Finally, we would like to mention an addition we've made to the paper which we believe strengthens it. An interesting facet of our proposed architecture is the **connection to graph neural networks and the advantages our approach has over GNNs for relational representation learning**. Although the current 8-page version of the manuscript contains some discussion of this, with the additional page allowance **we will expand upon this discussion in the final version**. This discussion is somewhat related to your concerns about the difference from spatial convolutions. Below is a draft of the discussion which we will add to the paper.

---

Graph neural networks can be understood through the lens of neural message-passing as follows: $h_i^{(l+1)} \gets \mathrm{Update}(h_i^{(l)}, \mathrm{Aggregate}(\{h_j^{(l)}, \ j \in \mathcal{N}(i)\}))$. That is, in a GNN, the hidden representation of node $i$ is updated as a function of the hidden representations of its neighbors on a graph. Here, the graph is part of the input to the model. This framework has proven powerful in tasks including node classification, graph classification, and link prediction. This type of model is often referred to as "relational" since the edges in the input graph can be thought of as relations.

This is in contrast to another line of work on "relational architectures," where rather than being given as input, the task-relevant relations must be inferred end-to-end [1,2,3,4,5]. Our work builds on this literature. The input in this case is merely a collection of objects, and the task requires reasoning about the relations between objects. In many situations, the task-relevant relations may be more complex higher-order relations. The goal of our work is to propose a natural architecture that can learn to capture such relations by composing basic architectural building blocks.

While GNNs can be applied to such relational tasks by passing a complete graph as input along with the collection of objects, a GNN will in general lack the relational inductive biases needed to infer the task-relevant relations. This can be seen from the general form of neural message-passing. When a complete graph is given as input, each object's representation is updated as a function of *all other objects' features*. There is no step where the relations between those features are explicitly computed. In other words, the "message" being sent from $\mathcal{N}(i)$ ($i$'s neighbors) to $i$ is an aggregation of the features of $i$'s neighbors. In a GNN, in general, the input graph is only used to direct the flow of information, but this information encodes object-level features, not relations. In fact, a Transformer can be viewed as an instantiation of a GNN operating on a complete graph. Notably, in the case of Transformers, the attention scores indirectly encode some relational information by weighting the edges of the graph. This confers certain modeling benefits but still lacks important relational inductive biases.

By contrast, in relational convolutional networks, relations are at the center of every operation. Pairwise relations are computed *explicitly* by the MD-IPR module. Then, the resultant relation tensor is convolved against a graphlet filter to produce representations of the relational pattern in groups of objects. Crucially, we emphasize that the convolution is *of the relations themselves* against the graphlet filters. This is in contrast to GNNs, where the convolution is of the objects *along* the edges of the graph. This enables relational convolutional networks to *infer* the task-relevant relations end-to-end, without requiring them to be received as input like GNNs do. Moreover, our architecture is able to learn representations of iteratively more complex relational features by composing simple building blocks.

---
[1] Santoro, et al. "A simple neural network module for relational reasoning." NeurIPS 2017

[2] Shanahan, et al. "An explicitly relational neural network architecture." ICML 2020

[3] Webb, et al. "Emergent Symbols through Binding in External Memory." ICLR 2020

[4] Kerg, et al. "On Neural Architecture Inductive Biases for Relational Tasks." ICLR 2022 Workshop OSC

[5] Altabaa, et al. "Abstractors and relational cross-attention: An inductive bias for explicit relational reasoning in Transformers." ICLR 2024

---

## Final remarks

Thanks again for engaging with our work. Please let us know if we have addressed your initial concerns and if you have any further comments or questions.

In particular, if we have addressed your confusion about the connection to spatial convolutions (weakness 1) and computational complexity (weakness 2), we would be very appreciative if you could update your score accordingly.