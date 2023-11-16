Thank you very much for your feedback and your thoughts. We have incorporated feedback from you and other reviewers and made several additions to the paper. We believe this meaningfully improves the paper, and are grateful for this feedback. Please see the global response and the updated pdf for these additions. We hope you will take a look. Thank you for your engagement.

---

> The authors claim the method to be more interpretable and parameter-efficient manner, but there is no analysis for the same.

Thanks for the suggestion! We have added a section in the appendix analyzing the representations learned by the MD-IPR and RelConv layers on the "contains set" task. This is an appendix C of the updated pdf. The findings are quite interesting. We observe that the encoders in MD-IPR learn to encode and separate the four latent attributes in the task (number, color, fill, and shape). Moreover, the learned graphlet filters process these relations to *exactly separate* triplets of objects which form a "set" from those that do not. Please see Figure 7-9 in the appendix.

We also added a discussion of parameter efficiency in appendix B.1. The key idea is that the number of parameters in MD-IPR and RelConv is independent of the number of objects and of the number of groups considered. This leads to computational and statistical benefits, as useful transformations are shared across pairs of objects and groups of objects. That is, similar to the translation invariance of CNNs, the inner product relations are shared across pairs of objects and the graphlet filters are shared across groups of objects.

---

Please take a look and let us know if this addresses your concern. We'd appreciate any further comments or questions you may have.