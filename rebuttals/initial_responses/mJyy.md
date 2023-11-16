We are grateful for your feedback. We have integrated it into an updated version of the paper, and believe that it has improved the final paper. Please see the global response, the updated pdf, and the responses below.

---

> Limited analysis of how the architecture scales as the number of objects and relations grow large. Memory and computation costs need investigation.

Thank you for the constructive suggestion! Indeed, it is possible to reduce the computation of the $n \times n$ relation tensor. Since the relational convolution considers relations *within a group*, we only need to compute relations between objects which co-occur in the same group. We have added an appendix section (B) which discusses how this can be done. Please also see the slightly-updated discussion on computing group-match scores in section 3.2. The key quantity is the set of discrete groups and group match scores. Then, a sparse relation tensor can be computed based on the "support" of those quantities. That is, $\mathcal{R} = \{R_{ij} : \exists\, g \in \mathcal{G} \ \text{such that} \ i,j \in g\}$ in the case of convolutions with fixed discrete groups, and $\mathcal{R} = \{R_{ij} : \exists\, k \in [n_g], g \in \mathcal{G} \ \text{such that} \ \alpha_{gk} > 0 \ \text{and} \ i,j \in g\}$ in the case of learned soft groups. By choosing a sparse normalizer in equation (7), $\alpha_{gk}$ will be sparse and hence $\mathcal{R}$ is sparse. For example sparsemax is used in the results added to the relational games experiments in section 5.1.

---

> Main experiments are on simple synthetic tasks.

We agree that more complex tasks, requiring modeling higher-order relations, would form an interesting evaluation of the architecture. Since existing relational architectures (e.g., PrediNet, CoRelNet, RelationalNet) are non-compositional, and did not focus on learning hierarchical relations, we were unable to find benchmarks for explicitly higher-order relational representation. The SET experiment is a step towards that, but we agree that evaluation on even higher-order relations as well as more realistic tasks would be interesting. The construction of such benchmarks requires careful thought and consideration, which was outside the scope of this project. However, we did add a section to the appendix (section D) which proposes some early ideas on tasks involving higher-order relations and potential real-world applications of the relational convolutions architecture.

---

A minor note:
> Experiments show the architecture is more sample efficient at relational reasoning tasks compared to Transformers and other baselines lacking explicit relational structure.

The PrediNet and CoRelNet architectures that we compare to are "explicitly relational architectures" from previous work on relational representation. You are likely aware of this, but we thought we'd clarify in case this point was missed. The difference is that those architectures lack the hierarchical/compositional structure of relational convolutional networks, making them less efficient on more difficult relational tasks (e.g., "match pattern" in the relational games), and completely unable to perform higher-order relational tasks (e.g., the SET experiments).

---
Thanks again for your feedback and engagement with our work. We hope that this answers some of your questions and addresses some of your concerns. Please let us know if you have any other questions or feedback.