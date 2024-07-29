Thank you for your review and constructive feedback.

We have uploaded a revised version of the manuscript with several additions to address feedback from reviews. Please see the global response for an overview of these additions. Below, we will address your particular concerns and highlight additions related to your review in more detail.

**Summary**
- We carried out a hyperparameter sweep for each baseline model searching over combinations of architectural hyperparameters and optimization hyperparameters.
- We updated the main text of the paper to compare the proposed RelConvNet architecture against optimized baselines. The RelConvNet model did not need hyperparameter tuning and continues to use the same default Adam optimizer.
- The hyperparameter sweep resulted in marginally improved performance for the baselines, but does not change any conclusions. In particular, RelConvNet remains the only model able to learn the difficult *Set* task.

**High-level concern about architecture and optimization hyperparameter choice**
> The high-level problem is that broad generalizations about entire classes of architectures based on one hyperparameter set for each architecture

The main high-level concern of this review is whether the conclusions drawn about the comparisons between our proposed architecture and the baseline architectures hold generally, or if they are specific to the specific architectural hyperparameters or optimization hyperparameters that we evaluate. We agree that it is important to validate whether the same conclusions continue to hold for different hyperparameter choices. In particular, the hyperparameter choice for each baseline should be approximately representative of what each model architecture is ideally capable of, for a given task.

To further validate our conclusions, we perform an extensive hyperparameter sweep over optimization hyperparameters (Adam vs AdamW, weight decay, learning rate schedule) and architectural hyperparameters (number of layers) *individually for each model*. We note that this was not necessary for RelConvNet; we continue to use the Adam optimizer with a fixed learning rate, no weight decay, and the TensorFlow default hyperparameters. Our aim is to compare against the best-achievable performance for each baseline model class, giving the baselines the advantage of hyperparameter tuning which was unnecessary for RelConvNet. Appendix B in the updated manuscript describes the hyperparameter tuning process carried out for each baseline model architecture and the different optimization hyperparameters chosen for each model. We re-ran each baseline with the optimal hyperparameters and updated the results in the main text of the paper.

The hyperparameter sweep resulted in marginally improved performance in the baselines, but did not change the general message of the experimental results. On the *Set* task, the best-performing model (excluding RelConvNet) without hyperparameter tuning was the LSTM achieving a generalization accuracy of 60.2%; the best-performing model after the hyperparameter sweep is a 1-layer GAT achieving a generalization accuracy of 67.5\%. For most models, the improvement from hyperparameter tuning was small. For the GAT baseline, however, the difference was significant, rising from 51.7% to 67.5%. RelConvNet remains by far the best-performing model, achieving a generalization accuracy of 97.9\% (without tuning optimization hyperparameters).

Below, we briefly summarize the results of the hyperparameter sweep with respect to the factors that you asked about.

**Hyperparameter sweep procedure for Baselines.**

We ran a total of 1620 experimental runs performing a hyperparameter sweep searching over combinations of architectural and optimization hyperparameters for all baselines with the goal of finding a hyperparameter configuration that is representative of the best achievable performance for each baseline. The results of this sweep are summarized in Appendix B. The full results and experimental logs of all runs will be made publicly available (through a W&B portal) in the de-anonymized final paper, same as the other experimental results from the paper.


**The effect of depth on performance in baselines.**
> In contrast to the Transformer model being compared to, which has a single layer in each setting, as to the GNN variants

> Perform a reasonable hyperparameter sweep for the experimental section, critically including multilayer variants of all architectures, include the results of the hyperparameter sweep as an Appendix section.

We vary the number of layers between 1 and 8 for each baseline. We find that increasing depth beyond two layers is generally detrimental. In the Transformer, GCN, and GIN the two-layer model slightly outperformed a one-layer model; in the GAT the one-layer model slightly outperformed the two-layer model. We choose the depth in the final model for each architecture accordingly.

**Optimization Hyperparameters**

> It is also unclear if standard optimization procedure is being followed

> Optimization should be done using optimizers that use weight decay sensibly (e.g. AdamW), and a learning rate schedule should be used. The optimal weight decay may be zero, and other values should be investigated.

As explained in the paper, we use the Adam optimizer with a learning rate of $0.001$, $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-7}$. These are commonly used settings and are the Tensorflow defaults for Adam. In the first version of the paper, the same optimization hyperparameters were used for all models and were not tuned for any model, making the comparison fair. Nonetheless, as you point out, it is possible that tuning these optimization hyperparameters might result in meaningful improvements in the baseline architectures.

We performed a hyperparameter sweep with the AdamW optimizer and used a weight decay ranging between 0.004 and 1.024. We also explored whether a learning rate schedule such as linear warmup + cosine decay improved results.

The detailed results are in the updated manuscript. Weight decay resulted in marginal improvements for some models, but no discernable improvement in others. The cosine learning rate schedule resulted in a significant improvement in the GAT model, but no improvement in the other baselines. The optimal weight decay level and learning rate schedule are chosen for each model according to these results.


These results support the conclusion that the reason for the strong overfitting of Transformers and GNNs is not the choice of optimization hyperparameters or limited depth, but rather the lack of appropriate inductive biases for learning complex relational representations. In particular, the *Set* task requires reasoning about relational patterns within groups of objects, which RelConvNet is able to explicitly learn and represent, whereas the other baselines cannot.


**On overfitting and relational inductive biases.**

Our results show that our initial conclusions broadly hold and that the baseline models' inability to solve the more complex relational tasks (e.g., *Set*) is not due to a poor choice of hyperparameters, but rather is due to a lack of appropriate relational inductive biases. For example, the combinatorial nature that *Set* exhibits as a relational reasoning task implies a hard limit on the number of training examples and requires the model to learn to generalize from limited data. In particular, it requires a particular type of relational reasoning involving hierarchical and compositional processing. Although deep models like Transformers or GAT are highly expressive and are able to fit a large class of datasets, they lack an inductive bias to discover this type relational reasoning procedure.

We highlight that this conclusion is broadly consistent with previous work tackling relational tasks (e.g., [1,2,3]) where it is found that Transformers under-perform. (These works don't evaluate against GNNs, but the same discussion applies.) This can be understood through the perspective of inductive biases. Since there exist many choices of parameters that fit the training data, we need *inductive biases* that help the trained model choose a sensible solution among these possible solutions. In the case of relational tasks, you need appropriate relational inductive biases. In our work, we particularly focus on *hierarchical* relations. 

<!-- This agrees with our experimental results where, for example, in the *Set* task, a Transformer is able to perfectly fit the training data but is unable to generalize. This is because it hasn't learned the right "procedure". -->

---
References:

[1] Shanahan, et al. "An explicitly relational neural network architecture." ICML 2020

[2] Webb, et al. "Emergent Symbols through Binding in External Memory." ICLR 2020

[3] Kerg, et al. "On Neural Architecture Inductive Biases for Relational Tasks." ICLR 2022 Workshop OSC


**Briefly, some minor points**

> Add the number of layers used for every architecture to the hyperparameter tables 2 and 4.

The number of layers is described in these tables. For example, "$\cdots \to (\mathtt{GCNConv} \to \mathtt{Dense}) \times 2 \to \cdots$" means that the GAT model has 2 layers.

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

You may or may not want symmetry over permutations of groups, depending on the task. For example, in the *Set* experiments of section 4.2 intuitively you would want symmetry, since "setness" is a permutation-invariant relational group property. By contrast, in the "match pattern" task, the property of having an AAA vs ABA vs AAB vs ABB vs ABC relation among triplets of objects is *not* permutation-invariant, and hence you would not want symmetry in your representation.

In our architecture, the "relational inner products" as defined in Equation 3 are not permutation-invariant by default (i.e., different permutations of objects can have different relational patterns computed by a relational convolution). In the subsection "Symmetric relational inner products" in Section 3.2, we discuss a symmetric variant of the relational inner product. This is used for the *Set!* experiments, but not the relational games experiments.