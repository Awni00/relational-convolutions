Dear reviewers,

Thank you for your constructive feedback. We have made several additions to the paper in response to your comments, and have made every effort to address each of the criticisms and concerns that were raised.  We will summarize the changes at a high-level in this global response and delve into more detail in the individual responses.

1. We have added an exploration of the effect of entropy regularization in group attention on training dynamics and model performance. We confirm that entropy regularization is necessary for escaping local minima at initialization. Higher regularization leads to lower entropy and sparser group attention scores, but good performance is achievable even with a small amount of regularization. [Appendix A, Figures 9 & 10]

2. We have added an exploration and discussion of the effect of the RelConvNet architecture hyperparameters on the ability to learn the challenging *Set* task. We find that multi-dimensional relations and a symmetry inductive bias are crucial for learning the task in a manner that generalizes. [Appendix A, Figure 11].

3. We added a CNN baseline to all experiments to explore a reviewer's question about whether a deep CNN could learn relational tasks end-to-end from raw image input. We find that the CNN model succeeds at learning the easier relational games tasks, but completely fails to learn the more challenging relational games tasks and the *Set* task. [Updated Section 4, Figures 4,5,7, Appendix A]

4. In our initial experiments, we used the same optimization hyperparameters for all models, without an individual hyperparameter sweep. A reviewer was appropriately critical of this setup. Based on the reviewer's suggestion, we carried out an extensive hyperparameter sweep over architectural hyperparameters and optimization hyperparameters for each baseline model individually. This makes it possible to compare against the best-achievable performance for each baseline model class, giving the baselines the advantage of hyperparameter tuning, which was unnecessary for RelConvNet. We find that some baselines benefited marginally from hyperparameter tuning (some more than others), but the message of our experimental results remains the same. In particular, RelConvNet remains the only model that can solve the *Set* task. [Description of hyperparameter sweep in Appendix B of updated manuscript; Results updated in Section 4]

Detailed descriptions of each of these changes to the paper are provided in the responses to the individual reviewers, where we also respond to each of the more minor comments that the reviewers made. We thank you again for helping to improve the quality of our work.


