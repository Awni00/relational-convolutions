# Reviewer 1 (xLCW)

<!-- [1 / n]: CONFUSION ON TERMINOLOGY, SETTING, & EXISTING LITERATURE -->

<!-- TODO: should we mention (more politely) that: the vast majority of the review is rambling about what they think "object" and "relations" should mean, without ever mentioning the technical contents of the paper -->
<!-- TODO: mention even more papers that use the same terminology of "objects" and "relations" -->
<!-- Should we write our rebuttal with the AC in mind, to try to demonstrate that this reviewer is unqualified? or should we try to reason with the reviewer? not sure if the latter is possible, but also don't know if the AC will take this into account -->
<!-- annoyingly this review is modestly long, giving the impression that there has been some thought put into this even though most of it is them rambling about their "first ML course" -->

## Clarifying confusion on terminology, the setting, and existing literature

Thank you for your review and your questions. We believe your concerns and confusion arise from a lack of familiarity with this specific area of research. In particular, there is confusion regarding basic terminology ("What do you assume an object is? What is a relationship?"). In fact, this terminology is standard in this line of work as we explain below.

For example, you talk about "Earth, cups or conferences, maybe all of the marriages" as objects in the first paragraph; and you refer to tabular datasets in the final paragraph "So it look just like a regular tabular dataset that we learned about in our first ML course".

This reflects some basic misunderstanding of the *setting* of our work which hinders evaluating our contributions and proposed methods. We are grateful for your willingness to engage in discussion with us to better-understand the setting, existing literature, and our proposed method.

> Pleas convince me that I am wrong because I think I am missing something.

We greatly appreciate your honesty and your willingness to engage in discussion. Below, we will clarify terminology, review the literature to help you understand the setting and existing work, and then summarize our contributions in this paper. We hope to clarify your confusion and address your concerns below in turn.


## Terminology & Existing Literature

- Our work builds on a line of work on "relational architectures" and "relational representation" [1,2,3,4, 5]. This is all cited in the introduction to our paper.
- **Our use of the terms "object" and "relation" is standard in this literature.** (e.g., see the references below.)
- **Our setting matches this existing literature** (i.e., the kinds of problems we are tackling). The data format and the nature of the tasks we tackle is the same as this literature.

Although you may be unfamiliar with this literature, we hope that we will be able to provide you with some of the background necessary to understand our work and provide an evaluation.

---
[1] Santoro, et al. "A simple neural network module for relational reasoning." NeurIPS 2017

[2] Shanahan, et al. "An explicitly relational neural network architecture." ICML 2020

[3] Webb, et al. "Emergent Symbols through Binding in External Memory." ICLR 2020

[4] Kerg, et al. "On Neural Architecture Inductive Biases for Relational Tasks." ICLR 2022 Workshop OSC

[5] Altabaa, et al. "Abstractors and relational cross-attention: An inductive bias for explicit relational reasoning in Transformers." ICLR 2024

---

<!-- [2 /n] DEFINING TERMINOLOGY AND EXAMPLES -->
## Defining terminology ("objects" & "relations") & some examples
Now, we give intuitive definitions of the terms "object" and "relation" as they are used in this literature, and go through some examples.

**Defining terminology in this literature**. An "object" is a vector representation of the features of an entity. A (pairwise) "relation" is a mapping from a pair of objects to a representation of the relations between their features. A relational model receives as input a collection of objects, and must reason about the relations between them in order to perform a task.

**Example 1 (Relational Games).** Consider the "relational games" task in the first set of experiments (Section 4.1). Here, the model must reason about the relations between the visual attributes of different objects in a visual scene. In particular, the scene is presented as a 3x3 grid of objects. In this setting, each cell in the grid is an "object" and each object is represented by an RGB image. Thus, the input to the model is a sequence of 9 images (the objects).

Note that each object varies across several feature attributes. In particular, their shape and color. Thus, in this example, a relation between two objects may be a mapping for the pair of images to a vector representing similarity across each of these attributes. The task is to learn to infer and represent these relations, then reason about these relations to perform the classification task.

**Example 2 (*SET!*).** In the second set of experiments on the *SET!* task, the "objects" are the different *SET!* cards (again, each represented as an RGB image in this case). Each "object" varies across several feature attributes (shape, color, fill, number). The task-relevant relations is whether each of these attributes is the same or different. To perform the task, the model must reason about the *relational pattern in a group of objects*.

**Other examples.** The framework of "objects" and "relations" as treated in this literature is very general and applies to a wide array of settings and tasks. For example, in addition to relations across visual features (as in the above examples), one may also consider relations in the context of language, where the objects are representations of words or sentences and the relations are semantic or syntactic. Please see the references we mentioned for additional examples if needed.

*Does this clarify your confusion about terminology and the setting? Please let us know if you have any further questions and we would be happy to clarify further.*

<!--  [3/n] BRIEF OVERVIEW OF METHODS IN EXISTING LITERATURE -->
## Brief overview of existing methods in the literature

In hopes of further explaining the setting addressed by our work and the context of our work within the literature, we will briefly describe the methods proposed in two related works. We hope that providing examples of methodologies proposed for this setting in addition to our own will help explain the setting.

[1] proposes the Relation Network (RN) module. The module receives as input a set of objects $X = (x_1, \ldots, x_n) \in \reals^{n \times d}$ where each $x_i \in \reals^d$ is an object whose features are represented as a vector. The module returns a processed vector which summarizes the relations between the objects as follows,
$$ \mathrm{RN}(X) = f(\sum_{i,j} g([x_i, x_j])),$$
where $g$ is an MLP, $[x_i, x_j]$ is the concatenation of the two vectors, and $f$ is another MLP. $g$ is intended to compute pairwise relations and $f$ processes the sum-aggregation of the pairwise relations to produce an output.

[2] proposes the CoRelNet architecture. It similarly receives a sequence of objects as input. CoRelNet simply computes the pairwise inner products of the objects, computes a softmax, then flattens this $n \times n$ matrix and passes it to an MLP. That is,
$$\mathrm{CoRelNet}(X) = \mathrm{MLP}(\mathrm{Flatten}(A)), \ A = \mathrm{Softmax}(X X^\top)$$

---
[1] Santoro, et al. "A simple neural network module for relational reasoning." NeurIPS 2017

[2] Kerg, et al. "On Neural Architecture Inductive Biases for Relational Tasks." arxiv:2206.05056, ICLR 2022 Workshop OSC

<!-- [4/n] Summary of our proposed method and contributions -->
## Summary of our proposed method and contributions

With the above clarification, we hope that you will be able to better-understand our proposed method and contributions. First, we summarize our method, then explain its significance. We defer technical details to the main text of the paper. We invite you to ask any questions if further clarification is needed.

1. The input to the model is a collection or sequence of "objects". Each object's features is represented by a vector.
2. Pairwise relations between object features are computed using the multi-dimensional inner product relations module. This is done by learning feature maps which extract or 'filter' attributes to be compared by an inner product. Multi-dimensional relations are enabled by multiple such feature maps. This produces a **relation tensor**.
3. A set of **graphlet filters** are learned which represent a template of the relational pattern within a group of objects. This template is matched against patches of the relation tensor through a novel **relational convolution** operation. This operation computes a *representation of the relational pattern in each group of objects*.
4. To avoid computing a convolution with a combinatorial number of groups, we propose a means of **learning the relevant groups end-to-end**. This is done through an attention-based operation, which retrieves objects for each group in a differentiable manner. This is a key technical challenge.
5. The relational convolution operation returns a sequence of new "objects" (i.e., a sequence of vectors), with each object now representing the relational pattern within a group of objects. This enables the overall operation to be repeated in order to **learn representations of higher-order relations**.

The main emphasis of our work is to design a **a natural, flexible, and compositional architecture for learning hierarchical relations**. This has **so far been unexplored in the literature**. In particular, relational architectures have so far been "flat", limiting their ability to learn iteratively more complex relational representations. Thus, our work forms an important contribution to the literature, addressing a central problem (learning hierarchical relations through a compositional architecture) which has so far been unaddressed.

<!-- [5/n] -->
## Clarification about question on notation in Eq (2)

Hopefully, you now have a better understanding of our work, the setting in which it operates, and the context of the current literature. We now turn to responding to your more specific questions.

> What does the notation of equation (2) mean? I think you want r(x,y)[k] = ... (i.e., the kth element of the vector).

$r(x, y)$ is a $d_r$-dimensional vector describing the relation between object $x$ and object $y$. If you're unfamiliar, the notation $[m]$ means $\{1, 2, \ldots, m\}$. Thus, the $k$-th element of the vector $r(x, y)$ (where $k \in [d_r]$) is given by $\langle W_1^{(k)} \phi(x), W_2^{(k)} \phi(y) \rangle$. The notation $( \cdot )_{k \in [d_r]}$ of course means the vector of elements in the parentheses where $k$ ranges from $1$ to $d_r$.

<!-- [6/n] representing multi-way relations -->
## On representing multi-way relations
> pairwise relations are not very rich. Don't think that your relational sub-tensors can represent 3-way relations

Indeed, pairwise relations are limited. **Our proposed method can in fact naturally represent multi-way relations between several objects.** This is achieved by our proposal of relational convolutions (and group attention). This ability to represent multi-way (and higher-order) relations is a key contribution of our work over the existing literature.

To see how relational convolutions can represent multi-way relations, consider the following example based the *SET!* experiments which rely on this type of higher-order relations. Recall that we observe in our experiments that our model is the only one which was able to learn the task, where other relational architectures previously proposed in the literature where completely unable to learn in a way that generalizes. This validates the need for architectures which explicitly learn representations of higher-order relations, such as relational convolutional networks.

**Example of $k$-way relations (*SET!*).** In the *SET!* task, the task-relevant information is the relational pattern between triplets of objects---that is, a 3-way relation. Recall that a "set" is a triplet of cards where across each of the 4 attributes, the cards are either all the same or all different. The property of being a "set" is a 3-way relational property. To represent such a property within the relational convolutions framework, you first compute the pairwise relations, which represent the same/different relation across each of the 4 attributes for each pair of objects. Then, you learn a *graphlet filter* which captures the all different/all same property. This captures the 3-way relation which would be needed to determine if a triplet of objects is a "set".

<!-- [7/n] -->
## Addition of discussion about connection to GNNs/Transformers

An interesting facet of our work (as well as the line of work it is a part of) is how it relates to graph neural networks---another kind of "relational" model. With the additional page allowance, **we will add an expanded discussion of this connection in the final version**. Below is a draft of the discussion which we will add to the paper.

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

We hope that you find this discussion interesting. We believe this addition meaningfully improves our work (particularly for readers less familiar with the literature).

<!-- [8/n] -->
## Final remarks
We hope that the clarification above has helped you understand our work and the setting in which it operates. Your initial review reflects some fundamental misunderstandings of our work and the literature it is a part of. We hope that the clarification above provides you with enough background and context to evaluate our work with the appropriate understanding. Please let us know if you have any further questions or comments. We look forward to hearing back from you.


---
<!-- IGNORE THE BELOW FOR NOW, CAN DISCUSS LATER IF NECESSARY -->
<!-- [?/n] COMMENT ABOUT OBJECT-CENTRIC REPRESENTATIONS AS INPUT -->
[Our proposed module receives object-centric representations as input. In some applications (e.g., image processing), object-centric representations may not be immediately available and would need to be generated by some other module within the network. There exists a long line of work tackling this exact problem. For example, Slot Attention [CITE], Complex Autoencoders [CITE], etc...]

Related but orthogonal problem.