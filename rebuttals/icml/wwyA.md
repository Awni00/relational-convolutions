# Reviewer 4 (wwyA)

Thank you for your review. We are glad you found our paper to be well-written and our experimental results to be good. We hope to address your concerns and make some clarifications.

Your review contains an important misunderstanding of the architecture:
>pairwise relation for the input is the "attention matrix", the paper then uses convolution network on top of the attention matrix.

This is cited as the reason for "novelty is limited" (weakness 1). In fact, this is not a correct description of our proposed architecture. The statement about the computational complexity (weakness 2) of our method is incorrect (perhaps as a result of this misunderstanding):
>I assume the total complexity is at least O(n^3) (n is input number), cannot scale.

The computational complexity scales *linearly* with the number of inputs.

We will explain both of these points in detail below, and explain our proposed method. We hope to clarify your confusion and look forward to further discussion with you!

## Understanding the relational convolutions architecture: correcting some misunderstandings

> Novelty is limited and some part is confusion. pairwise relation for the input is the "attention matrix", the paper then uses convolution network on top of the attention matrix. why choose convolution on this level of reasoning? the inductive bias for convolution is locality. however, there is no this type of this attributes on pairwise relation obtained by MD-IPR layer in the model.

**Our proposed architecture is *not* to apply a convolutional network on top of the attention matrix**. As you mention, the inductive bias for convolution is spatial locality, which does not make sense as a way to process the pairwise relations produced by the MD-IPR layer. There may be some confusion arising as a result of the word "convolution" which of course bears resemblance to 2D spatial convolutions. The **"relational convolution" operation is in fact a novel operation we define which is distinct from the spatial convolution operation in CNNs**. In particular, **in relational convolutions, the convolution is with respect to *groups*, not spatial translations as in the 2D convolutions of CNNs.**

We will explain the proposed architecture below in more detail and emphasize how it differs from the convolutions in traditional CNNs for image processing.

### High-level overview

First, we summarize the architecture at a high level.

1. The input to the model is a collection or sequence of "objects". Each object's features is represented by a vector.
2. Pairwise relations between object features are computed using the multi-dimensional inner product relations module. This is done by learning feature maps which extract or 'filter' attributes to be compared by an inner product. Multi-dimensional relations are enabled by multiple such feature maps. This produces a **relation tensor**.
3. A set of **graphlet filters** are learned which represent a template of the relational pattern within a group of objects. This template is matched against patches of the relation tensor through a novel **relational convolution** operation. This operation computes a *representation of the relational pattern in each group of objects*.
4. To avoid computing a convolution with a combinatorial number of groups, we propose a means of **learning the relevant groups end-to-end**. This is done through an attention-based operation, which retrieves objects for each group in a differentiable manner. This is a key technical challenge which we address.
5. The relational convolution operation returns a sequence of new "objects" (i.e., a sequence of vectors), with each object now representing the relational pattern within a group of objects. This enables the overall operation to be repeated in order to **learn representations of *higher-order* relations**.

Now, we describe the relational convolution operation in more technical detail. We will draw the contrast to standard 2D convolutions for both relational convolutions with discrete groups and relational convolutions with group attention.

### Relational Convolutions with Discrete Groups

In this case, we consider a setting where a collection of discrete groups $\mathcal{G}$ is provided as input to the model. The first step is to compute the $n \times n \times d_r$ relation tensor using the MD-IPR layer.

We learn $n_f$ different graphlet filters, each of size $s \times s \times d_r$, where $s$ is the size of the graphlet (number of objects in each group). These act as templates of the relational pattern in a group of objects. $n_f$ and $s$ are both hyperparameters.

**The relational convolution operation compares the pairwise relations within groups of $s$ objects against the graphlet filters**. Given a group $g \subset [n]$ of $s$ objects, the pairwise relations within this group are given by the relation subtensor $R[g] := [R[i,j]]_{i,j \in g}$. The relation subtensor is compared against the graphlet filters $\bm{f} = (f_1, \ldots, f_{n_f}) \in \reals^{n_f \times s \times s \times d_r}$ to produce an $n_f$-dimensional vector summarizing the relational pattern in group $g$ (see Eqs 3,4 in main text).

The graphlet filters are convolved *across groups* $g \in \mathcal{G}$ to produce as sequence of $|\mathcal{G}|$ objects as follows,
$$R * \bm{f} := (\langle R[g], \bm{f}\rangle_{\mathrm{rel}})_{g \in \mathcal{G}}$$

**In a relational convolution operation, the convolution is across *groups*, not spatial translations as in the 2D convolutions of CNNs.** In particular, $\bm{f}$ is matched against the groups in $\mathcal{G}$, whereas in a 2D Convolution, $\bm{f}$ would be matched against *contiguous* patches of the relation tensor (e.g., $[0:s], [1:s+1], [2:s+2], ..., [n-s:n]$, etc.). As you point out, this would not make sense in the context of relational processing. **To emphasize this contrast to the reader, and highlight the novelty of our approach, we will add a brief discussion to the paper explaining the difference between relational convolutions and 2D convolutions.**

### Relational Convolutions with Group Attention

In some situations, the set of task-relevant discrete groups may not be available and considering all possible groups may be impractical. To address this setting, we propose a variant of relational convolutions where groups are learned and represented differentiably via an attention operation. This is a major contribution of our work. The contrast with 2D convolutions remains of course.

Below, we will summarize the methodology. Please see section 3.2 of the paper for a complete description.

1. First, for each of $n_g$ groups (a hyperparameter) we retrieve $s$ objects from the collection of $n$ objects. We denote these objects by $\bar{x}_i^g$, where $g \in [n_g], i \in [s]$. (See Eq 7)
2. Next, we compute the relation tensor for each group $g \in [n_g]$ using a shared MD-IPR layer producing $\bar{R}[g] \in \reals^{s \times s \times d_r}$. (See Eq 8)
3. For each group $g$, $\bar{R}[g]$ is compared against the graphlet filters $\bm{f}$ producing the relational convolution $\bar{R} * \bm{f}$, which consists of a sequence of $n_g$ vectors, summarizing the relational pattern within each learned group. (See Eq 9)

**This bears little resemblance to 2D convolutions and introduces several architectural innovations.**

Does this address your concern about the connection between our architecture and standard 2D convolutions applied to the "attention scores"? Please let us know if you have further questions. We look forward to your response.

### MD-IPR vs Attention Scores

Finally, a minor comment about multi-dimensional inner product relations vs attention scores. Indeed, the MD-IPR layer bears a resemblance to the attention matrix of Transformers. A couple of clarifications are in order. This is not an "attention" matrix because it is not used to "retrieve" anything (i.e., in a Transformer, attention computes a convex combination of objects in the sequence weighted by the attention scores). That is, the attention matrix is an intermediate step used to direct the flow of object-level features. In MD-IPR, the relation tensor is the *final output* of the MD-IPR layer and is used directly as a representation of the relations. Related to this, the relation tensor is not normalized via a softmax as in attention.

## Computational Efficiency & Scalability


As explained above, our architecture is not to "use [a] convolutional network on top of the attention matrix" and hence **the conclusion about computational complexity and scalability in the review is incorrect** (weakness 2). We discuss computational complexity at the end of Section 3.2. The **computational complexity is in fact linear in $n$**, the number of objects in the input.

The overall computational complexity of one layer in a relational convolutional network only scales linearly with the number of objects in the input. **In section 3.2, we discuss the computational complexity of each step in the relational convolution layer**. In particular, the group attention operation (Eq 7) has complexity $O(n \cdot n_g \cdot s \cdot d)$, where $n$ is the number of objects in the input. Computing the relation tensors for each group $\bar{R}[g]$ (Eq 8) has complexity $O(n_g \cdot s^2 \cdot d_r \cdot d_{\mathrm{proj}}). Finally, the complexity of computing the relational convolution (Eq 9) is $O(n_g \cdot s^2 \cdot d_r \cdot n_f)$, where $n_f$ is the number of graphlet filters (also a hyperparameter). Note that *all of these operations can be computed in parallel independently* using well-optimized GPU operations. Moreover, the first operation (group attention) scales only linearly with input size, and the latter two scale only with the model size (i.e., hyperparameters) and not the size of the input.


----

## Contrast to GNNs

Related to your concern about the connection to 2D convolutions, is the **connection to graph neural networks and the advantages our approach has over GNNs for relational representation learning**. Althout the current 8-page version of the manuscript contains some discussion of this, with the additional page allowance **we will expand upon this discussion in the final version**. Below is a draft of the discussion which we will add to the paper.

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

## Final remarks

Thanks again for engaging with our work. Please let us know if we have addressed your initial concerns and if you have any further comments or questions.

In particular, if we have addressed your confusion about the connection to 2D convolutions (weakness 1) and computational complexity (weakness 2), we would be very appreciative if you could update your score accordingly.

