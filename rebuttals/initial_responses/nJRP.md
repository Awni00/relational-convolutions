We appreciate your feedback. Please see the global response as well as the updated pdf for a description of some of the additions we made based on the feedback of reviewers. Below are some responses to your questions.

---

> Some techinical details need to be claimed

Can you clarify what you mean by this?

---
> More explanations about Fig.2 should be included, like symbol defination, and how you process pooling on graph.

Thanks for the suggestion. We have expanded the description in the caption. Please note that the notation is defined in the main body of the text, as well as in the annotations in the figure. The first row of the figure depict the graphlet filters which correspond to the parameters $\bm{f} \in \mathbb{R}^{s \times s \times d_r \times n_f}$. In this example, the filter size $s$ is 3. In the second row, the relation tensor $R$ is depicted as a graph with 5 nodes, and the red highlights correspond to each relation subtensor of $s=3$ objects---$g \in \mathcal{G}$ denotes a subset of $s$ objects, and $R[g]$ denotes the subtensor of $R$ corresponding to those objects (as defined in section 3.1 of the text). For each $g \in \mathcal{G}$, $R[g]$ is compared against the filter $\bm{f}$ via the relational inner product $\langle \cdot, \cdot \rangle_{\mathrm{rel}}$, as defined in equations 3 and 4. This yields the relational convolution $R \ast \bm{f}$ as defined in equation 6. Finally, given a group matrix $G \in \mathbb{R}^{m \times n_g}$, the relational convolution *given* $G$ is computed as described in equations 7-9.

---

> In relation convolution layer, can you introduce the movitation of the design of attention-based pooling and more clear explanation about the techinical details?

Are you asking about the context-aware grouping of equation 11? Here, the proposal is to generate a group matrix $G$ which assigns objects to soft groups using not only the object's own features, but also based on the context in which the objects appear. In general, this can be done via a message-passing neural network, as described in equation (11), by obtaining an embedding $E_i \gets \mathrm{MessagePassing}(x_i, \{x_1, \ldots, x_m\})$ for each object. The message-passing operation can learn to incorporate the relevant information about object $i$'s context in $E_i$ such that $E_i$ can be mapped to a vector $\mathrm{MLP}(E_i) \in \reals^{n_g}$ encoding the object's group membership as a function of its context within the sequence. This contextual information may include information about the features of other objects in the sequence as well as their relation with object $i$.

In the experiments, we use a Transformer Encoder block to encode the context (i.e., via self-attention). This is among the additions in the updated pdf. Please see section 5.1, Figure 5, and Table 3 in the appendix.

---

> In experiments, do you treat each pix of image as an token input for transformer? or a patch like ViT?

The input to the Transformer is a sequence of $m$ vector embeddings of each object image, obtained via a CNN. The sequence of objects are $(x_1, \ldots, x_m)$ where each $x_i \in \mathbb{R}^{w \times h \times 3}$ is an RGB image. Each image is processed independently using a CNN that produces a vector embedding $z_i \in \mathbb{R}^d$. The vector embeddings $(z_1, \ldots, z_m)$ are passed as input to the Transformer, just as for the other architectures. The CNN Embedder has a common architecture and the details are described in the appendix (Table 2 and Table 5, respectively, for each set of experiments).
