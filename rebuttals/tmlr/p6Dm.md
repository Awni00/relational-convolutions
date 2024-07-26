Thank you for your review.

We have uploaded a revised version of the manuscript with several additions to address feedback from reviews. Please see the global response for an overview of these additions. Below, we will address your particular concerns and highlight additions related to your review in more detail.

> Since the method involves several modules, ablation studies on the contribution of each module would be beneficial.

We have updated the paper to add some discussion of the effects of different configurations of the proposed module. In particular, the effect of the relation dimension $d_r$, imposed symmetry of the relations, entropy regularization, and others. For brevity, here we will highlight what we thought was most interesting.

Recall that the *Set* task was by far the most difficult relational task in our experiments, with all other baselines failing to learn the task beyond 60\% accuracy. Here, we discuss what factors contribute to the success of relational convolutional networks on the task.

We find that the symmetry of relations is a crucial inductive bias. We experiment with variants of RelConvNet with symmetric ($W_1 = W_2$) and asymmetric ($W_1, W_2$ independent parameters) relations. We find that RelConvNet fails to learn the *SET!* task in a way that generalizes without the symmetry constraint---it is able to fit the training data but does not learn the rule in a way that generalizes.

We also find have multi-dimensional relations to be crucial for reliably learning this task. We compare a RelConvNet model with $d_r = 1$ to one with $d_r = 16$ (the variant reported in the paper). Interestingly, we find that in most trials, the $d_r = 1$ model gets stuck at around 50\% accuracy, but in a few trials it does manage to learn the task reaching near-perfect accuracy. We have two intuitive explanations for this. The first is that this task of course relies on reasoning about relations across four different attributes. The second is that having multiple relations (i.e., $d_r > 1$) gives the model multiple different avenues to find a solution, each initialized from a different starting point. When $d_r = 1$, you may be unlucky and get stuck at a local minimum with no path to a good solution. But with multiple relations, a model is able to explore multiple avenues towards a solution making it much more likely to find a good one.

Part of what makes this task very challenging is the need to perform some kind of combinatorial search with a limited supervision signal. This manifests itself in the shape of the accuracy curve over the curse over the course of training. It exhibits a "staircase" shape, suggesting that, when the model successfully learns the task, it does so all at once. We find that having symmetric and multi-dimensional relations is crucial for learning such tasks that require hierarchical combinatorial-like relational reasoning tasks.

These new results are depicted in Figure 11 in the appendix.

> In the introduction, it is helpful to give a clear explanation of the originality of the work over existing ones (e.g., with respect to the problem setting considered or learning method).

We will make a pass over the introduction to try to emphasize the originality of our work and its relation to existing work. Below is some discussion on this.

Our work is falls within a line of recent work concerned with developing neural network architectures with relational inductive biases for relational reasoning tasks. Representative examples of work on this problem include [1,2,3,4], and have influenced our thinking over the course of the project.

In our work, the problem we tackle is developing neural architectures for learning *hierarchical* relational representations. Here, "hierarchical" means relations between relations. Previous work on relational architectures has been shallow and limited to first-order pairwise relations. Recognizing that deep learning's success stems from composing (simple) modules together to build iteratively more complex feature maps, we set out to develop a compositional relational architecture with the ability to learn hierarchical, higher-order relations.

The way we propose doing this is through a novel operation that can be thought of as analogous to convolutions in a CNN---we call it *relational convolutions*. Here, the key idea is to learn *graphlet filters* which represent templates of relational patterns between groups of objects and to "convolve" this with against the relations between the input. This produces a sequence of embeddings, each representing the relational pattern in some grouping of objects. Repeating the operation produces higher-order representations of relations-between-relations.

---

References:

[1] Santoro, et al. "A simple neural network module for relational reasoning." NeurIPS 2017

[2] Shanahan, et al. "An explicitly relational neural network architecture." ICML 2020

[3] Webb, et al. "Emergent Symbols through Binding in External Memory." ICLR 2020

[4] Kerg, et al. "On Neural Architecture Inductive Biases for Relational Tasks." ICLR 2022 Workshop OSC