# Reviewer 3 (ucGQ)

Thank you for your thoughtful review. We appreciate your positive feedback. 

You note the lack of experiments on real-world data and the limited explanation about advantages over GNNs as the two primary weaknesses of this paper. We appreciate your comments. We hope to address your concerns below and look forward to further discussions with you.

## Expanded discussion on inductive bias and advantage over GNNs and Transformers

> The authors discussed GNNs in related work, but did not illustrate the advantages of the proposed method compared with GNNs. Although GNNs rely on explicitly given ‘relations’ but they can be applied into this setting by view data as a complete graph (which is also acknowledged by the authors). The advantages of RelConvNet vs GNNs should be explained more explicitly.

Thank you for the question and suggestion. We agree that an expanded discussion about the advantages of the proposed method over GNNs would be an interesting addition to strengthen the paper. The current 8-page version of the manuscript contains a short discussion of this (related work section (L078-097) and the experiments section (L404-433)). With the additional page allowance, **we will expand upon this discussion in the final version**. Below is a draft of the discussion which we will add to the paper.

---
Graph neural networks can be understood through the lens of neural message-passing as follows: $h_i^{(l+1)} \gets \mathrm{Update}(h_i^{(l)}, \mathrm{Aggregate}(\{h_j^{(l)}, \ j \in \mathcal{N}(i)\}))$. That is, in a GNN, the hidden representation of node $i$ is updated as a function of the hidden representations of its neighbors on a graph. Here, the graph is part of the input to the model. This framework has proven powerful in tasks including node classification, graph classification, and link prediction. This type of model is often referred to as "relational" since the edges in the input graph can be thought of as relations.

This is in contrast to another line of work on "relational architectures," where rather than being given as input, the task-relevant relations must be inferred end-to-end [1,2,3,4,5]. Our work builds on this literature. The input in this case is merely a collection of objects, and the task requires reasoning about the relations between objects. In many situations, the task-relevant relations may be more complex higher-order relations. The goal of our work is to propose a natural architecture that can learn to capture such relations by composing basic architectural building blocks.

While GNNs can be applied to such relational tasks by passing a complete graph as input along with the collection of objects, a GNN will in general lack the relational inductive biases needed to infer the task-relevant relations. This can be seen from the general form of neural message-passing. When a complete graph is given as input, each object's representation is updated as a function of *all other objects' features*. There is no step where the relations between those features are explicitly computed. In other words, the "message" being sent from $\mathcal{N}(i)$ ($i$'s neighbors) to $i$ is an aggregation of the features of $i$'s neighbors. In a GNN, in general, the input graph is only used to direct the flow of information, but this information encodes object-level features, not relations. In fact, a Transformer can be viewed as an instantiation of a GNN operating on a complete graph. Notably, in the case of Transformers, the attention scores indirectly encode some relational information by weighting the edges of the graph. This confers certain modeling benefits but still lacks important relational inductive biases.

By contrast, in relational convolutional networks, relations are at the center of every operation. Pairwise relations are computed *explicitly* by the MD-IPR module. Then, the resultant relation tensor is convolved against a graphlet filter to produce representations of the relational pattern in groups of objects. Crucially, we emphasize that the convolution is *of the relations themselves* against the graphlet filters. This is in contrast to GNNs, where the convolution is of the objects *along* the edges of the graph. This enables relational convolutional networks to *infer* the task-relevant relations end-to-end, without requiring them to be received as input like GNNs do. Moreover, our architecture is able to learn representations of iteratively more complex relational features by composing simple building blocks.


---
References:

[1] Santoro, et al. "A simple neural network module for relational reasoning." NeurIPS 2017

[2] Shanahan, et al. "An explicitly relational neural network architecture." ICML 2020

[3] Webb, et al. "Emergent Symbols through Binding in External Memory." ICLR 2020

[4] Kerg, et al. "On Neural Architecture Inductive Biases for Relational Tasks." ICLR 2022 Workshop OSC

[5] Altabaa, et al. "Abstractors and relational cross-attention: An inductive bias for explicit relational reasoning in Transformers." ICLR 2024

---

We hope that you find this discussion interesting. We thank you again for the suggestion. Does this address weakness (2) in your review? Please let us know if you have any further questions or comments.

## Complexity of computing pairwise inner products of feature maps

Thank you for the question. We discuss the computational complexity of each step of our proposed architecture at the end of Section 3.2 (L215-242) in our paper. The complexity of computing the relation subtensors via the pairwise inner products (the step in Eq 8) is $O(n_g \cdot s^2 \cdot d_r \cdot d_{\mathrm{proj}})$. Note that these are all hyperparameters of the model and do not scale with the input size. In particular, the hyperparameter $s$ is likely to be chosen as some fixed small number such as $3$ or $5$. $n_g$ is the number of learned groups and its value should be specified depending on the task. $d_r$ and $d_{\mathrm{proj}}$ are again hyperparameters (relation dimension and projection dimension). Overall, the computational complexity depends only on the scale of the model and not the scale of the input.

In case you are interested, the overall computational complexity of one layer in a relational convolutional network is also discussed in Section 3.2 and only scales linearly with the number of objects in the input. In particular, the group attention operation (Eq 7) has complexity $O(n \cdot n_g \cdot s \cdot d)$, where $n$ is the number of objects in the input, and the complexity of computing the relational convolution (Eq 9) is $O(n_g \cdot s^2 \cdot d_r \cdot n_f)$, where $n_f$ is the number of graphlet filters (also a hyperparameter). Note that all of these operations can be computed in parallel independently using well-optimized GPU operations. Thus, our proposed method is computationally-efficient. 

## Experiments on real-world data

> As acknowledged by the authros, only synthetic experiments are conducted. Since the paper is not a theoretical paper, it would be better to have some experiments on real-world data (e.g., data in computer vision and RL) in order to sufficiently showcase the effectiveness the approach.

We completely agree that experiments with real-world data would be important to further demonstrate the effectiveness of our approach, and we plan to address this in future work. As you mentioned, we discuss this under "limitations and future work" in the conclusions section.

In addition, in case you missed it, we would like to draw your attention to Appendix C in the paper which discusses some ideas on real-world higher-order relational tasks which can be pursued by future work.

We would like to take this opportunity to provide a brief discussion of these ideas. While this does not fully resolve this limitation of our work, we hope that it makes for some interesting discussion for future directions. We would be happy to incorporate a version of this discussion into the main text of the paper with the additional space allowed in the final version.

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

-----

Thank you once again for the well-considered review. Please let us know if we have addressed your initial concerns and if you have any further comments or questions. If we have addressed your concerns, we would be very appreciative if you could update your score accordingly.