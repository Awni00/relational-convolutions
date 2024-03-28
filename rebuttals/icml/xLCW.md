# Reviewer 1 (xLCW)

<!-- [1 / n]: CONFUSION ON TERMINOLOGY, SETTING, & EXISTING LITERATURE -->

Thank you for your review. This review centers around confusion on two basic terms used in our paper and the associated literature: "object" and "relation". *We kindly emphasize that this terminology is standard in this line of work* (e.g., see the following key works [1,2,3,4,5]). We believe this review lacks the basic understanding of the setting of this line of work which would be necessary to provide an informative evaluation. Unfortunately, this confusion has resulted in several fundamental inaccuracies and mischaracterizations, such as the assertion that our work is related to tabular data, which is incorrect. Furthermore, the review lacks a discussion of the technical content of our paper, such as the proposed methodology and experimental results.

We understand that reviewers have varying backgrounds and levels of expertise in different areas of research. We also understand that if the reviewer did not have an understanding of the basic setting they would be unable to evaluate the technical content. We will aim to constructively engage with the reviewer to help them understand this line of work better. To address these issues, our rebuttal will be structured as follows:
1. Clarification on the terminology used in this line of work
2. An introductory overview of existing methods in this literature
3. A summary and reminder of the proposed methodology and contributions of this paper

We hope this will clarify the reviewer's confusion about the basic setting and enable them to engage more meaningfully with our work. We remain open to further discussions and clarifications.

---
References:

[1] Santoro, et al. "A simple neural network module for relational reasoning." NeurIPS 2017

[2] Shanahan, et al. "An explicitly relational neural network architecture." ICML 2020

[3] Webb, et al. "Emergent Symbols through Binding in External Memory." ICLR 2020

[4] Kerg, et al. "On Neural Architecture Inductive Biases for Relational Tasks." ICLR 2022 Workshop OSC

[5] Altabaa, et al. "Abstractors and relational cross-attention: An inductive bias for explicit relational reasoning in Transformers." ICLR 2024

---

## Clarifying confusion around the basic terminology

As explained above, the majority of the text of the review centers around confusion on the terms "object" and "relation". For example, the opening reads,
> I think of a list of all of the people on Earth, cups or conferences, maybe all of the marriages. (Are people, cups, conferences or marriages objects? If not, then what are objects?) You seem to only input a list the objects; so I presume we just give a unique number to each person. That can't be correct, as you need properties of the objects, but there are no relations input (e.g, which company made the cup)

Since we do not understand the text of your review (it does not seem to be related to the setting we tackle), we will aim to provide a general introductory overview of this line of work.

As explained above, these terms are standard in the literature [1-5]. Moreover, the experiments in our paper give concrete examples of the types of problems being tackled by this area of research. We note that this literature is cited and discussed in the paper. The setting of our work matches this literature. That is, the data format and the nature of the tasks we tackle are the same. This line of work tackles problems in which a machine learning model must reason about a collection of objects, inferring relations between them in order to perform the task.

We invite the reviewer to look at the related literature to gain an improved understanding of this setting. For example, [1] applies their proposed method to reasoning about visual features of objects in a scene. [2] proposes the "relational games" suite of benchmark tasks, which we also use to evaluate our methods in this paper. [3] focuses on visual relational reasoning tasks based on cognitive tests. [4] tests their proposal on the benchmarks used in [2] and [3]. [5] tests their proposal on sequence modeling tasks such as "object-sorting" and mathematical problem-solving.

"Relational reasoning" is a central component of generally intelligent behavior and has applications in a wide range of tasks.

In the next section, we will explain the terminology of "object" and "relation" from the perspective of its representation in a neural architecture.

<!-- [2 /n] DEFINING TERMINOLOGY AND EXAMPLES -->
## Explaining terminology ("objects" & "relations") & some examples

**Defining terminology in this literature**. An "object" is a vector representation of the features of an entity that the model must reason about in relation to other entities. A (pairwise) "relation" is a mapping from a pair of objects to a representation of the relations between their features. A relational model receives as input a collection of objects, and must reason about the relations between them in order to perform a task.

**Example 1 (Relational Games).** Consider the "relational games" task in the first set of experiments (Section 4.1). Recall that this suite of benchmark tasks is used in [2,4] as well. In these tasks, the model must reason about the relations between the visual attributes of different objects in a visual scene. In particular, the scene is presented as a 3x3 grid of objects. In this setting, each cell in the grid is an "object" and each object is represented by an RGB image. Thus, the input to the model is a sequence of 9 images (the objects).

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

1. The input to the model is a collection or sequence of "objects". Each object's features are represented by a vector.
2. Pairwise relations between object features are computed using the multi-dimensional inner product relations (MD-IPR) module. This is achieved by learning feature maps that extract or 'filter' attributes for comparison through an inner product. Multiple feature maps enable the consideration of multi-dimensional relations. This produces a **relation tensor**.
3. A set of **graphlet filters** are learned which represent a template of the relational pattern within a group of objects. This template is matched against patches of the relation tensor through a novel **relational convolution** operation. This operation computes a *representation of the relational pattern in each group of objects*.
4. We propose an attention-based operation, which we call **"group attention"**, to *learn the relevant groupings end-to-end*. Group attention retrieves the relevant objects for each group *differentiably*. This approach alleviates the need to consider a combinatorial number of groups during the computation of the relational convolution, addressing a key technical challenge.
5. The **relational convolution** operation returns a sequence of new "objects" (i.e., a sequence of vectors), with each object now representing the relational pattern within a group of input objects. This enables the overall operation to be repeated in order to *learn representations of higher-order relations*.


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

<!-- Reviewer doesn't ask about this; they may not be able to understand it. we can skip it for this reviewer?  -->
<!-- [7/n] -->
<!-- ## Addition of discussion about connection to GNNs/Transformers

An interesting facet of our work (as well as the line of work it is a part of) is how it relates to graph neural networks---another kind of "relational" model. With the additional page allowance, **we will add an expanded discussion of this connection in the final version**. Below is a draft of the discussion which we will add to the paper.

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

We hope that you find this discussion interesting. We believe this addition meaningfully improves our work (particularly for readers less familiar with the literature). -->

<!-- [8/n] -->
## Final remarks
We hope that the clarification above has helped you understand our work and the setting in which it operates. Your initial review reflects some fundamental misunderstandings of our work and the literature it is a part of. We hope that the clarification above provides you with enough background and context to engage with our work with the appropriate understanding. Please let us know if you have any further questions or comments. We look forward to hearing back from you.

---
<!-- IGNORE THE BELOW FOR NOW, CAN DISCUSS LATER IF NECESSARY -->
<!-- [?/n] COMMENT ABOUT OBJECT-CENTRIC REPRESENTATIONS AS INPUT -->
<!-- [Our proposed module receives object-centric representations as input. In some applications (e.g., image processing), object-centric representations may not be immediately available and would need to be generated by some other module within the network. There exists a long line of work tackling this exact problem. For example, Slot Attention [CITE], Complex Autoencoders [CITE], etc...]

Related but orthogonal problem. -->