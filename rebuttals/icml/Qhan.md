# Reviewer 2 (Qhan)

Thank you for your review and for engaging with our work. We are glad you found our proposal innovative and hope you got something out of it. We hope to address your concerns below and look forward to further discussion!

## Expanded discussion on inductive bias and advantage over GNNs and Transformers

> Could the authors provide a more concise formal discussion regarding the enhanced capacity or improved generalizability over transformer or GNN architectures? Specifically, how do relational convolutional networks represent hierarchical relations more efficiently compared to transformers?

Thank you for the question and the suggestion. We agree that this would significantly strengthen the paper. Accordingly, with the additional page allowed for the final submission, **we will add a discussion section that specifically addresses how our model relates to GNNs/Transformers and what advantages it has over these models**.

Indeed, the connection to and advantages over GNNs and Transformers is interesting because GNNs are usually described as "relational" models. However, as we explain below, they operate in a different domain where the "relations" are assumed to be received as input to the model in the form of a graph. By contrast, our model (and the line of work we build on) must *infer* the task-relevant relations and learn relational representations in an end-to-end fashion.

Below, we provide a draft of the discussion which will be added to the paper.

---

Graph neural networks can be understood through the lens of neural message-passing as follows: $h_i^{(l+1)} \gets \mathrm{Update}(h_i^{(l)}, \mathrm{Aggregate}(\{h_j^{(l)}, \ j \in \mathcal{N}(i)\}))$. That is, in a GNN, the hidden representation of node $i$ is updated as a function of the hidden representations of its neighbors on a graph. Here, the graph is part of the input to the model. This framework has proven powerful in tasks including node classification, graph classification, and link prediction. This type of model is often referred to as "relational" since the edges in the input graph can be thought of as relations.

This is in contrast to another line of work on "relational architectures," where rather than being given as input, the task-relevant relations must be inferred end-to-end [1,2,3,4,5]. Our work builds on this literature. The input in this case is merely a collection of objects, and the task requires reasoning about the relations between objects. In many situations, the task-relevant relations may be more complex higher-order relations. The goal of our work is to propose a natural architecture that can learn to capture such relations by composing basic architectural building blocks.

While GNNs can be applied to such relational tasks by passing a complete graph as input along with the collection of objects, a GNN will in general lack the relational inductive biases needed to infer the task-relevant relations. This can be seen from the general form of neural message-passing. When a complete graph is given as input, each object's representation is updated as a function of *all other objects' features*. There is no step where the relations between those features are explicitly computed. In other words, the "message" being sent from $\mathcal{N}(i)$ ($i$'s neighbors) to $i$ is an aggregation of the features of $i$'s neighbors. In a GNN, in general, the input graph is only used to direct the flow of information, but this information encodes object-level features, not relations. In fact, a Transformer can be viewed as an instantiation of a GNN operating on a complete graph. Notably, in the case of Transformers, the attention scores indirectly encode some relational information by weighting the edges of the graph. This confers certain modeling benefits but still lacks important relational inductive biases.

By contrast, in relational convolutional networks, relations are at the center of every operation. Pairwise relations are computed *explicitly* by the MD-IPR module. Then, the resultant relation tensor is convolved against a graphlet filter to produce representations of the relational pattern in groups of objects. Crucially, we emphasize that the convolution is *of the relations themselves* against the graphlet filters. This is in contrast to GNNs, where the convolution is of the objects *along* the edges of the graph. This enables relational convolutional networks to *infer* the task-relevant relations end-to-end, without requiring them to be received as input like GNNs do. Moreover, our architecture is able to learn representations of iteratively more complex relational features by composing simple building blocks.

---
We hope that you find this discussion interesting. We thank you again for the question and suggestion; we believe this addition has meaningfully improved our paper. Does this address your concern? Please let us know if you have any further questions or comments.


## Discussion on *real-world* applications

> Can you elaborate on potential real-world applications or scenarios where relational convolutional networks could be applied beyond the experimental tasks considered in the paper?

We agree about the importance of this question. In the current version of the paper, **we discussed real-world applications in Appendix C**. This discussion needed to be relegated to the appendix in the initial version of the paper due to space constraints. The final submission allows an additional page, which will allow us to **add a discussion of real-world applications in the main text**. Please find a draft of the discussion to be added below.

---

While the experiments in this paper are primarily synthetic, allowing for a more controlled evaluation of the proposed architecture, there exist several important real-world tasks where relational convolutional networks could be applied.

One example is *computer vision and visual scene understanding*. In any naturalistic visual scene, there are typically several objects and the spatial, visual, and semantic relations between them are crucial for parsing the scene. The CLEVR benchmark on visual scene understanding [1] was used in early work on relational representation [2]. In more complex situations, the objects in the scene may fall into natural groupings, and the spatial, visual, and semantic relations between those groups may
be important for parsing a scene (e.g., objects forming larger components with functional dependence determined by the relations between them). Integrating relational convolutions into a visual scene understanding system may enable reasoning about such higher-order relations. An important task that is related but orthogonal to relational processing is "object discovery". A promising direction of future work would be to explore how object-discovery methods such as Slot Attention [3] can be incorporated with relational processing modules such as relational convolutional networks.

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