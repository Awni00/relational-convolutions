We are thrilled you found our paper interesting and enjoyable to read! Thank you for your thoughts, feedback, and suggestions. We have incorporated your feedback into an updated version of the paper, and we think they form interesting additions. Please see the global response, updated pdf, and the response below for more details.

---

> Despite interpretability...

We agree that this would be interesting to see! We added a section to the appendix (section C) exploring and visualizing the representations learned by the MD-IPR and RelConv layers on the SET task. The results are interesting---we indeed see that different encoders in the MD-IPR layer learn to extract and compare different choices of the latent attributes in the data (i.e., number, color, fill, and shape). We also find that the RelConv layer learns graphlet filters which exactly separate triplets of cards which form 'sets' from those that don't! We hope you find this added analysis interesting.


---

> (This point is also mentioned ...

We entirely agree. The strength of this architectural framework is its ability to model hierarchical relations, but our experiments were limited to synthetic tasks relying on relations of first or second-order. While working on this project, we struggled to find benchmark tasks that fully test this ability. As you said, we found ourselves with "a solution in search of a problem," or more precisely, in search of a *benchmark*. We believe the ability to represent and reason about hierarchical relations has the potential to be useful in several interesting applications, including sequence modeling, set embedding, and visual scene understanding. We agree that the paper would benefit from more of a discussion on potential applications. We added an appendix section (section D) which discusses some ideas about applications of higher-order relational representation, and possible benchmarks that could be explored in future work.

---

> In Table 2 of the appendix, ...

Good question! Your intuition is correct that each additional layer of MD-IPR+RelConv models relations of one degree higher. In the case of the "match pattern'' task, the relations are indeed second-order, but the second-order relation is relatively simple. Having extracted the first-order relation among the triplet of objects, the MLP needs only to determine whether those first-order relations are the same or different. Thus, while a second layer of MD-IPR+RelConv would work here as well, it is not necessary.

One way to think about this is that, given a sequence of objects, an MD-IPR layer computes first-order relations. Then, following a relational convolution layer, we obtain a sequence of objects, each summarizing the relational pattern within some group. Then, a second MD-IPR layer computes the relations between those relational objects, forming second-order relations. The RelConv layer in-between can be thought of as computing relations of an order that is part-way between first-order and second-order, because it has already grouped objects and computed the relational pattern between them.

Because the final second-order relation is very simple (same or different), an MLP can parse the output of the RelConv layer into the final prediction.

One noteworthy thing here is that, with the addition of learned grouping, RelConvNet can perform even better on the generalization split of the "match pattern" task in the relational games benchmark (see Figure 5 of the updated pdf where we have now added these results).

---

> I can think of ...

Parameter-efficiency is a big part of the reason. $\mathcal{G}$ will typically be exponential in the number of objects (e.g., of size $\binom{n}{s}$). This may be prohibitively large (computationally) when $n$ is moderately large. Another reason is the "statistical" aspect of how easy it would be to learn reasonable groups. We think this inductive bias makes learning meaningful groups easier because it links possible groups in $\mathcal{G}$, through object membership within them, rather than considering each group independently. In the case of temporal grouping, for example, group membership is determined by the temporal (positional) order in which the object occurs. It might be much more difficult to learn this kind of pattern with a parameterization in terms of $|\mathcal{G}|$ independent discrete groups instead. The $m \times n_g$ structure of the group matrix enables natural versions of the feature-based grouping and context-aware grouping layers as well, where an object's group assignment is simply a learned function of its features (i.e., $\phi(i, x_i) \in \reals^{n_g}$ is a vector representing the degree to which the object belongs to each group).

---

> What are some ...

Please see Appendix D of the updated pdf :)

---
> Could you provide some insight ...

Yes! Please see Appendix C of the updated pdf.

---

Please let us know if you have any thoughts about the additions or if you have any further questions or comments.