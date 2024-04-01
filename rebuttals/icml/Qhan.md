# Reviewer 2 (Qhan)

Thank you for your review and for engaging with our work. We are glad you found our proposal "innovative" and found the formalism of graphlet filters and relational convolutions to be "a clear and structured approach to modeling relational patterns and capturing higher-order relations". We really appreciate your positive feedback.

We also appreciate your questions and constructive criticism. Below, we aim to address your concerns and look forward to further discussion.

## Expanded discussion on inductive bias and advantage over GNNs and Transformers

> Could the authors provide a more concise formal discussion regarding the enhanced capacity or improved generalizability over transformer or GNN architectures? Specifically, how do relational convolutional networks represent hierarchical relations more efficiently compared to transformers?

We agree that an expanded discussion about the advantages of the proposed method over GNNs would be an important addition to strengthen the paper. The current 8-page version of the manuscript contains a short discussion of this (related work section (L078-097) and the experiments section (L404-433)). **We will expand upon this discussion in the final version of the paper**, taking advantage of the additional page allowance.

Below, we provide a draft of the discussion which will be incorporated into the final version of the paper.

---

Graph neural networks can be understood through the lens of neural message-passing as follows: $h_i^{(l+1)} \gets \mathrm{Update}(h_i^{(l)}, \mathrm{Aggregate}(\{h_j^{(l)}, \ j \in \mathcal{N}(i)\}))$. That is, in a GNN, the hidden representation of node $i$ is updated as a function of the hidden representations of its neighbors on a graph. Here, the graph is part of the input to the model. This framework has proven powerful in tasks including node classification, graph classification, and link prediction. This type of model is often referred to as "relational" since the edges in the input graph can be thought of as relations.

This is in contrast to another line of work on "relational architectures," where rather than being given as input, the task-relevant relations must be inferred end-to-end [1,2,3,4,5]. Our work builds on this literature. The input in this case is merely a collection of objects, and the task requires reasoning about the relations between objects. In many situations, the task-relevant relations may be more complex higher-order relations. The goal of our work is to propose a natural architecture that can learn to capture such relations by composing basic architectural building blocks.

While GNNs can be applied to such relational tasks by passing a complete graph as input along with the collection of objects, a GNN will in general lack the relational inductive biases needed to infer the task-relevant relations. This can be seen from the general form of neural message-passing. When a complete graph is given as input, each object's representation is updated as a function of *all other objects' features*. There is no step where the relations between those features are explicitly computed. In other words, the "message" being sent from $\mathcal{N}(i)$ ($i$'s neighbors) to $i$ is an aggregation of the features of $i$'s neighbors. In a GNN, in general, the input graph is only used to direct the flow of information, but this information encodes object-level features, not relations. In fact, a Transformer can be viewed as an instantiation of a GNN operating on a complete graph. Notably, in the case of Transformers, the attention scores indirectly encode some relational information by weighting the edges of the graph. This confers certain modeling benefits but still lacks important relational inductive biases.

By contrast, in relational convolutional networks, relations are at the center of every operation. Pairwise relations are computed *explicitly* by the MD-IPR module. Then, the resultant relation tensor is convolved against a graphlet filter to produce representations of the relational pattern in groups of objects. Crucially, we emphasize that the convolution is *of the relations themselves* against the graphlet filters. This is in contrast to GNNs, where the convolution is of the objects *along* the edges of the graph. This enables relational convolutional networks to *infer* the task-relevant relations end-to-end, without requiring them to be received as input like GNNs do. Moreover, our architecture is able to learn representations of iteratively more complex relational features by composing simple building blocks.

---
We hope that you find this discussion interesting. We thank you again for the question and suggestion; we believe this addition has meaningfully improved our paper. Does this address your concern? Please let us know if you have any further questions or comments.


## Discussion on *real-world* applications

> Weakness Lack of Real-World Applications: While the paper demonstrates the effectiveness of relational convolutional networks in capturing hierarchical relations on relational games, it lacks exploration or discussion of potential real-world applications beyond synthetic games.

> Can you elaborate on potential real-world applications or scenarios where relational convolutional networks could be applied beyond the experimental tasks considered in the paper?

We agree that the lack of discussion or exploration of potential real-world applications is a limitation of our work. We would like to highlight Appendix C in the paper which discusses potential real-world applications involving higher-order relational tasks which can be pursued by future work.

We would like to take this opportunity to provide a brief discussion on these ideas. We would be happy to incorporate a version of this discussion into the main text of the paper using the additional space allowed in the final version.

---

While the experiments in this paper are primarily synthetic, allowing for a more controlled evaluation of the proposed architecture, there exist several important real-world tasks where relational convolutional networks could be applied.

One example is *computer vision and visual scene understanding*. In any naturalistic visual scene, there are typically several objects and the spatial, visual, and semantic relations between them are crucial for parsing the scene. The CLEVR benchmark on visual scene understanding [1] was used in early work on relational representation [2]. In more complex situations, the objects in the scene may fall into natural groupings, and the spatial, visual, and semantic relations between those groups may be important for parsing a scene (e.g., objects forming larger components with functional dependence determined by the relations between them). Integrating relational convolutions into a visual scene understanding system may enable reasoning about such higher-order relations. An important task that is related but orthogonal to relational processing is "object discovery". A promising direction of future work would be to explore how object-discovery methods such as Slot Attention [3] can be incorporated with relational processing modules such as relational convolutional networks.

Another example is *sequence modeling* (e.g., *language modeling*). Modeling the relations between objects is usually essential for many sequence modeling tasks. For example, syntactic and semantic relations between words are crucial to parsing language. Higher-order relations are also important, capturing syntactic and semantic relational features across different locations in the text and multiple length-scales and layers of hierarchy. See for example some relevant work in linguistics [4,5]. The attention matrix in Transformers can be thought of as implicitly representing relations between tokens. It is possible that composing Transformer layers also learns hierarchical relations. However, as shown in this work and previous work on relational representation, Transformers have limited efficiency in representing relations. Thus, incorporating relational convolutions into Transformer-based sequence models may yield meaningful improvements in the relational aspects of sequence modeling. One way to do this is by cross-attending to the sequence of relational objects produced by relational convolutions, each of which summarizes the relations within a group of objects at some level of hierarchy.

---
[1] Johnson, et al. "CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning", CCVPR 2017

[2] Santoro, et al. "A simple neural network module for relational reasoning." NeurIPS 2017

[3] Locatello, et al. "Object-centric learning with slot attention." NeurIPS 2020

[4] Rosario, et al. "The descent of hierarchy, and selection in relational semantics." Assoc for Comp Linguistics 2002

[5] Frank, et al. "How hierarchical is language use?" Proc of Royal Society 2012

---

## Final remarks

Thank you again for your thoughtful review and for engaging with our work---we believe that it has meaningfully improved our paper.

Has the addition of the discussion on the comparison to GNNs/Transformers and real-world applications addressed your questions/concerns? If so, we would be very appreciative if you could update your scores accordingly. We look forward to discussing any further comments or questions you may have.

## Follow up question: Representation Capacity or Sample Efficiency

> To be precise, there could be two benefits
> 1. your inductive bias helps you to represent rich relational patterns that transformers can not attain. E.g., could come up with a relational pattern that can not be represented by transformers and can be represented by your method.
> 2. Your method requires less data than transformers to learn the same relational patterns.
> Which of the statements above is true and could you prove any of these in a formal or semi-formal way in your paper.

## Response to follow-up question: Representation Capacity or Sample Efficiency

### Intro / high-level summary

Thank you for your response and for the question. This is a great question. Both representational capacity and sample-efficiency play a role. The key, however, is the types of solutions that gradient-based training will be biased towards.

As you point out, the effects of different architectural inductive biases can be understood through several different lenses:
1. Representational capacity: What function class can an architecture represent, and how large a network is necessary and/or sufficient to represent a given function within a certain approximation error?
2. Learning dynamics: What kinds of solutions will the architecture find in practice through gradient-based training? How much data is needed to find good solutions (sample efficiency)?
3. Generalization: Will solutions found by gradient-based training generalize?

These questions are separate but related. Our claim about the inductive biases of our method can be explained in terms of these lenses. First, we summarize our claim in one short paragraph, then explain it in detail.

*A large enough Transformer can in principle approximate rich relational patterns since, at the end of the day, these are just sequence functions and Transformers are universal approximators. However, this says nothing about the ability of a model to **find** good relational representations through gradient-based training. Our method has inductive biases which bias it towards natural relational representations which enable gradient-based training to find good solutions in a way that is: **1) more sample-efficient, and 2) better able to generalize.***

### More in-depth explanation

Transformers (a similar discussion holds for other architectures such as GNNs) have good approximation capabilities. In particular, Transformers are universal approximators of permutation-equivariant sequence-to-sequence functions [1,2,3]. However, such results only show the *existence* of Transformers which approximate a given function. This type of analysis does not say anything about: 1) whether these types of solutions will be found by gradient-based training, 2) how much data would be necessary to find good solutions, or 3) whether the gradient-trained network will generalize out-of-sample or out-of-distribution.

Indeed, previous work on relational inductive biases has found that Transformers tend to be sample-*in*efficient and have limited generalization ability on tasks that rely on relational reasoning [4,5,6,7,8]. Our experimental results are consistent with this general trend. We find that our method *is more sample-efficient* and can *generalize* better compared to both Transformers and graph neural networks.

Our focus in this paper is on the representation of more complex hierarchical relations through composable architectural building blocks. This has so far been missing in the literature on relational inductive biases since previous work has only considered "flat" 1-layer architectures. Thus, we contrast our proposal to existing work in two ways: 1) deep models such as Transformers are able to build iteratively more complex feature maps by composition but lack relational inductive biases; and 2) relational models such as CoRelNet/PrediNet/etc have relational inductive biases, but lack compositionality. By contrast, relational convolutional networks are both compositional and have relational inductive biases. In particular, the relational inductive biases of RelConvNet tackle the challenging problem of learning hierarchical relations by explicitly learning to group objects and compute representations of the relational patterns within groups.

Below, will outline the empirical evidence for the improvement in sample-efficiency and generalization, then explain how the architectural choices and inductive biases yield these improvements.

---
References:

[1] Yun, et al. "Are Transformers universal approximators of sequence-to-sequence functions?" ICLR 2020

[2] Takakura & Suzuki. "Approximation and Estimation Ability of Transformers for Sequence-to-Sequence Functions with Infinite Dimensional Input" ICML 2023

[3] Alberti, et al. "Sumformer: Universal Approximation for Efficient Transformers". 2023

[4] Santoro, et al. "A simple neural network module for relational reasoning." NeurIPS 2017

[5] Shanahan, et al. "An explicitly relational neural network architecture." ICML 2020

[6] Webb, et al. "Emergent Symbols through Binding in External Memory." ICLR 2020

[7] Kerg, et al. "On Neural Architecture Inductive Biases for Relational Tasks." ICLR 2022 Workshop OSC

[8] Altabaa, et al. "Abstractors and relational cross-attention: An inductive bias for explicit relational reasoning in Transformers." ICLR 2024

### Empirical evidence

#### Sample-efficiency

In our work, we use training curves as a proxy to evaluate sample-efficiency. For example, on the "relational games" suite of benchmark tasks, Figure 4 shows the accuracy for each model against the number of training batches. We observe that RelConvNet is always at the top of these curves, learning the fastest of any model. The Transformer consistently requires the largest number of batches to learn the task. The effect is more pronounced on the "match pattern" task, which is a more complicated relational task involving a second-order relation.

#### Generalization

The relational games suite of benchmarks is designed to evaluate out-of-distribution generalization. In particular, the training setup is such that we train on one set of objects, and then evaluate on visually different sets of objects. Thus, the models must learn relational representations which are robust and invariant to the object space in order to generalize out-of-distribution. Figure 5 shows the out-of-distribution generalization performance of each model on each hold-out object set. We observe that RelConvNet generalizes out of distribution better than any other model on most task-split combinations, and that the effect is again pronounced on the more complex "match pattern" task.

On *SET!*, the benefit in generalization is even more extreme. This task is a complex higher-order relational tasks which requires reasoning about the relational patterns in groups of objects. We observe that *RelConvNet is the only model able to learn the task in a way which generalizes out-of-sample non-trivially*. In particular, RelConvNet *learns the rule* and achieves near-perfect accuracy (mean: $98\%$; std err: $1\%$). By contrast, the out-of-sample (i.e., test set) performance of the other tested models (Transformer, CoRelNet, PrediNet, GNNs) is near random chance.

Figure 9 shows the training accuracy and validation accuracy over the course of training. We observe that Transformers and GNNs are able to fit the data. However, they fail to learn the rule in a way that generalizes out-of-sample (i.e., on the hold-out test set). This can be understood through the distinction between broad representational capacity and specific inductive biases for sample-efficiency and generalization in a particular class of tasks: Transformers and GNNs are universal approximators and can fit the training data, but they lack the inductive biases to find the "right" solution to the underlying problem via gradient-based training. By contrast, RelConvNet is built of architectural blocks that *explicitly* represent hierarchical relational patterns.

In appendix D, we study the geometry of learned representations by RelCovNet to validate that the model *learns the "right" solution*. In particular, Figure 12 shows a visualization of the output of the relational convolution operation on all possible triplets of objects. We see that even when projecting to only two dimensions, the representation *sharply separates* the two classes. Thus, in a sense, the model learns the correct rule or program for solving the task. It reasons about the relational pattern in triplets of objects and, for each, it determines whether this triplet is a "set" or not. We emphasize that this kind of group-wise reasoning is not inevitable even for a model which achieves good classification performance since the visualization in Figure 12 is of the representation learned by an intermediate layer not the penultimate layer, and concerns subsets of the input not the full input.

### Understanding how RelConvNet's inductive biases achieve this

#### Formal theory
> could you prove any of these in a formal or semi-formal way in your paper.

The improvements in sample-efficiency and generalization are based on the complicated training dynamics of deep models like Transformers. Thus, a formal proof is out of scope for this paper. In fact, a formal analysis of the nonconvex optimization dynamics of Transformers is out of reach for the current theoretical tools available for analyzing deep learning models. So far, theoretical analysis of the optimization behavior of Transformers has been relatively limited in scope, typically assuming a toy model that only resembles a Transformer (e.g., 1-layer model with only attention and a linear decoder) [9,10,11]. Instead, in our work, we aim to understand this behavior empirically through our experiments.

#### Intuitive understanding & heuristic arguments

In our initial response, we provided a discussion of the inductive biases of our method compared to Transformers and GNNs under the heading "Expanded discussion on inductive bias and advantage over GNNs and Transformers". There, we provided a draft of an expanded discussion which will be added to the paper. It may be useful to refer to that discussion with the above in mind. Below, we will also provide a heuristic argument which may help give you some intuition about why would expect the particular inductive biases to have the observed effect on sample efficiency and generalization.

Recall that GNNs and Transformers can be understood through the lens of neural message-passing. In both Transformers and GNNs, the interaction between different objects is through sharing an objects features with another object. For example, in a GNN, each object receives a "message" from its neighbors encoding their features. With a Transformer, this is similar, except that attention weighs the relative contribution of each neighbor. At this point, the updated representation of each object encodes a mixed representation of the features of neighboring objects. Computing a *relation* between this mixed representation of features must be done by the MLP. However, the MLP has no restriction to compute relational representations, it is equally biased towards any function on this mixed representation.

By contrast, relational convolutional networks are composed of architectural blocks that *explicitly* compute relational representations. That is, gradient-based training will search over representations that are restricted to be within the space of *relational* representations. In particular, the MD-IPR layer explicitly computes pairwise relations by learning attribute filters and comparing attributes by inner products (i.e., the pair of object features is not mixed needing an MLP to disentangle them, like with Transformers/GNNs). Then, the Relational Convolution layer explicitly convolves a graphlet filter against the pairwise relations in order to produce representations of the relational patterns within groups. By composing these operations, *the architecture naturally captures hierarchical relations*.

For simple first-order relations, an architecture which lacks relational inductive biases (e.g., Transformer) might be able to overcome this limitation given a large enough amount of data. However, for more complex higher-order relations, this becomes much more difficult and no amount of data may be sufficient. It becomes much more likely that the architecture will memorize the training data without learning the rule. This is what we observe in our experiments.

---
References:

[9] Tarzanagh, et al. "Max-Margin Token Selection in Attention Mechanism." NeurIPS 2023

[10] Wu, et al. "On the Convergence of Encoder-only Shallow Transformers." NeurIPS 2023

[11] Vasudeva, et al. "Implicit Bias and Fast Convergence Rates for Self-attention" 2024

---

Thank you for your question and for engaging with our work! We appreciate the time that you are putting into this.

Does this help clarify your question? We hope this gives you a better understanding of the architectural inductive biases of our proposed method and how it is able to outperform models such as Transformers and GNNs on certain relational tasks. Please let us know if you have any further questions and we would be vary happy to provide more discussion.

If you believe we have adequately addressed your/questions concerns we would be very appreciative if you considered updating your score.