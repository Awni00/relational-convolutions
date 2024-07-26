Thank you for your review and feedback.

We have uploaded a revised version of the manuscript with several additions to address feedback from reviews. Please see the global response for an overview of these additions. Below, we will address your particular concerns and highlight additions related to your review in more detail.

**High-level concern about architecture and optimization hyperparameter choice**
> The high-level problem is that broad generalizations about entire classes of architectures based on one hyperparameter set for each architecture

The main high-level concern of this review is whether the conclusions drawn about the comparisons between our proposed architecture and the baseline architectures hold generally, or if they are specific to the specific architectural hyperparameters or optimization hyperparameters that we evaluate. This, of course, is a fair concern and applies to all experimental machine learning research which compares different architectures. While it is computationally infeasible to perform an exhaustive search over all combinations of architectural and optimization hyperparameters, we entirely agree that it is important to validate the domain over which one architecture is better than others with respect to hyperparameter choice in order to make the comparison fair. In particular, the hyperparameter choice for each baseline should be approximately representative of what each model architecture is capable of.

We have carried out additional experiments to explore the effects of model architecture hyperparameters and optimization hyperparameters on model performance for different baselines. In particular, we focus on number of layers (depth), weight decay, and learning rate schedule, following your suggestions. Our results show that our initial conclusions broadly hold and that the baseline models' inability to solve the more complex relational tasks (e.g., *Set*) is not due to a poor choice of hyperparameters, but rather is due to a lack of appropriate relational inductive biases.

We highlight that this conclusion is broadly consistent with previous work tackling relational tasks (e.g., [1,2,3]) where it is found that Transformers under-perform. (These works don't evaluate against GNNs, but the same discussion applies.) This can be understood through the perspective of inductive biases. Since there exists many choices of parameters that fits the training data, we need *inductive biases* that help the trained model choose a sensible solution among these possible solutions. In the case of relational tasks, you need appropriate relational inductive biases. In our work, we particularly focus on *hierarchical* relations. 

<!-- This agrees with our experimental results where, for example, in the *Set* task, a Transformer is able to perfectly fit the training data but is unable to generalize. This is because it hasn't learned the right "procedure". -->

---
References:

[1] Shanahan, et al. "An explicitly relational neural network architecture." ICML 2020

[2] Webb, et al. "Emergent Symbols through Binding in External Memory." ICLR 2020

[3] Kerg, et al. "On Neural Architecture Inductive Biases for Relational Tasks." ICLR 2022 Workshop OSC

**The effect of depth on performance in baselines.**
> In contrast to the Transformer model being compared to, which has a single layer in each setting, as to the GNN variants

> Perform a reasonable hyperparameter sweep for the experimental section, critically including multilayer variants of all architectures, include the results of the hyperparameter sweep as an Appendix section.

As you point out, the Transformer model in our baselines is a single-layer. We have carried out additional experiments to evaluate the performance of multi-layer Transformers. We did the same for all GNN baselines as well.

We varied the number of layers from 1 to 8 for each of the Transformer, GCN, GAT, and GIN baselines.

TODO: add description of results

...

**Optimization Hyperparameters**

> It is also unclear if standard optimization procedure is being followed

> Optimization should be done using optimizers that use weight decay sensibly (e.g. AdamW), and a learning rate schedule should be used. The optimal weight decay may be zero, and other values should be investigated.

As explained in the paper, we use the Adam optimizer with a learning rate of $0.001$, $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-7}$. These are commonly used settings. The same optimization hyperparameters are used for all models and are not tuned for any model. Thus, the comparison is fair.

Nonetheless, to address your concern about the effect of optimization hyperparameters learning rate schedules and weight decay, we carried out experiments to explore whether different optimizer settings can significantly change the results. In particular, we experimented with the AdamW optimizer and using a weight decay between 0.004 and 1.024. We also explored whether a learning rate schedule such as linear warmup + cosine decay improved results.

We found that this wasn't the case. In particular, recall that on the challenging *Set* task, no model other than RelConvNet was able to generalize above 60\% accuracy. This remained to be the case regardless of weight decay, learning rate schedule, or model depth.

[TODO: write description of results]

These results support the conclusion that the reason for the strong overfitting of Transformers and GNNs is not the choice of optimization hyperparameters or limited depth, but rather the lack of appropriate inductive biases for learning complex relational representations. In particular, the *Set* task requires reasoning about relational patterns within groups of objects, which RelConvNet is able to explicitly learning and represent, whereas the other baselines cannot.

**Briefly, some minor points**

> Add the number of layers used for every architecture to the hyperparameter tables 2 and 4.

The number of layers are in these tables. For example, "$\cdots \to (\mathtt{GATConv} \to \mathtt{Dense}) \times 2 \to \cdots$" means that the GAT model has 2 layers.

<!-- 
> Additionally it is known that for 12 layer transformer models, relational NLP tasks like Named Entity Recognition are solved very well (https://arxiv.org/abs/1810.04805). This can be done in 6 layers with distillation procedures (https://arxiv.org/abs/1910.01108).

Named entity recognition is an unrelated to

[They mentioned NER several times for some reason. This is an unrelated problem. How to talk to this point? I don't know what they're thinking.

One approach:
We tackle "relational tasks" in the sense the term is used in [1-6]. There is of course no formal definition or standard categorization of tasks. But NER falls outside the realm of focus for our work and [1-6]. ...
]
  -->

> For example, when talking about building pairwise relations, it could be useful to link to the transformer paper https://arxiv.org/abs/1706.03762

> Where discussing the proposal to model pairwise relations between objects, explain why Transformer and GNNs are insufficient, as both of these models naively sound like they would do this.

We'd like to refer you to the following paragraphs in the paper:
1. "It is perhaps surprising that models like GNNs and Transformers perform poorly on these relational tasks, given their apparent ability to process relations through ..." in Section 4.2
2. The "Discussion on relational inductive biases" subsection in the discussion.

We discuss how models like Transformers and GNNs differ from models like RelConvNet/PrediNet/CoRelNet/etc in the way they process relations. For example, in Transformers, the pairwise relations are used only as an intermediate step in an information retrieval operation rather than manifesting in the resultant representations. By contrast, relational architectures like ours produce a relation-centric representation.

> Equation 3: the i, j indices at this part of the manuscript feel unusual since there should be a symmetry over i, j as the filter acts on elements of a set.  ...

You may or may not want symmetry over permutations of groups. For example, in the *Set!* experiments of section 4.2 intuitively you would want symmetry, since "setness" is a permutation-invariant relational group property. By contrast, in the "match pattern" task, the property of having an AAA vs ABA vs AAB vs ABB vs ABC relation among triplets of objects is *not* permutation-invariant, and hence you would not want symmetry in your representation.

In our architecture, the "relational inner products" as defined in Equation 3 are not permutation-invariant by default (i.e., different permutations of objects can have different relational patterns computed by a relational convolution). In the subsection "Symmetric relational inner products" in Section 3.2, we discuss a symmetric variant of the relational inner product. This is used for the *Set!* experiments, but not the relational games experiments.