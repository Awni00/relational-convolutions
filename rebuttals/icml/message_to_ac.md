Dear AC,

We would like to express our concern about the review by reviewer xLCW which we believe is uninformative. The majority of the review focuses heavily on the reviewer's confusion about two basic terms used in our paper, "objects" and "relations". We emphasize that this terminology is standard in this line of work (e.g., see [1,2,3,4,5]), which we cite and discuss in the paper. *The text of the review never mentions our proposed method or experimental results, focusing solely on questioning basic terminology.* Moreover, the reviewer's comments have an unprofessional and confrontational tone.

This confusion on terminology is apparent throughout the review. For example, the first paragraph reads
>  I think of a list of all of the people on Earth, cups or conferences, maybe all of the marriages. (Are people, cups, conferences or marriages objects? If not, then what are objects?) You seem to only input a list the objects; so I presume we just give a unique number to each person. That can't be correct, as you need properties of the objects, but there are no relations input (e.g, which company made the cup). For some people there is lots known about them, and for some there is little. Some cups have a long history and some are brand new (but maybe that isn't part of the object). How can you determine relations between people (friendship, mentor, etc)? Your examples use what seems like bizarre notions of objects -- if your theory model only refers to these sorts of "objects" you should say so up-front.

The review demonstrates a lack of basic understanding of our paper's setting and the relevant literature, offering assertive but content-lacking criticism. While we understand that reviewers may vary in their familiarity with specific areas of research, we believe that a reviewer with a general knowledge of machine learning should be able to grasp the fundamental concepts by reading the paper. It is worth noting that all other reviewers demonstrate an understanding of the setting and terminology, even if some misunderstood certain technical aspects of our proposed methods.

The reviewer's lack of grasp of the basic setting has led to very fundamental mischaracterizations in the review. For example, the reviewer says:
> So it look just like a regular tabular dataset that we learned about in our first ML course.

Our work has nothing to do with tabular data, and this assertion again indicates a lack of understanding of the most basic elements of our work and the relevant literature. Indeed, the text of the review never mentions our proposed methodology or our experiments. As a side note, we find the phrase "we learned about in our first ML course" bizarre and raises concerns about the reviewer's qualifications to assess papers at a conference of this caliber.

We believe that this review does not provide an informative evaluation of our work. Moreover, it makes it difficult for us to respond to any apparent concerns since the majority of the review is incomprehensible to us.

Nevertheless, we would be happy to engage with this reviewer. In our rebuttal, we will aim to 1) clarify confusion about the terminology, explaining that this terminology is standard in this line of work; 2) provide a brief overview of some of the relevant literature to give the reviewer the needed background; and 3) summarize our proposed method and contributions.

However, we wanted to raise this issue to your attention early so that you can make an informed decision. Given these issues, we respectfully request that this review be discarded. Please let us know if you have any questions and we would be very happy to elaborate on our position further. We sincerely thank you for your consideration and your efforts in maintaining the quality of the review process.


---
References:

[1] Santoro, et al. "A simple neural network module for relational reasoning." NeurIPS 2017

[2] Shanahan, et al. "An explicitly relational neural network architecture." ICML 2020

[3] Webb, et al. "Emergent Symbols through Binding in External Memory." ICLR 2020

[4] Kerg, et al. "On Neural Architecture Inductive Biases for Relational Tasks." ICLR 2022 Workshop OSC

[5] Altabaa, et al. "Abstractors and relational cross-attention: An inductive bias for explicit relational reasoning in Transformers." ICLR 2024

---

<!-- ICML email -->
ICML email says the following:

> Area Chairs and Senior Area Chairs have been asked to read through reviews and work with Reviewers to ensure that they are high quality. However, with a conference the scale of ICML, it is unavoidable that some low quality preliminary reviews will slip through. If you receive a review that is inaccurate, disrespectful, or does not provide a meaningful assessment of your work, please send a confidential comment to the Area Chair handling your paper. You can use the “Official Comment” button to submit it. Note you cannot edit it after submitting, so proofread it carefully. Also, please ensure that the Readers field contains the Area Chair but not the Reviewers. (The readers field must also include the Senior Area Chair and Program Chairs.)


> Use “Official comment” to talk directly with ACs or SACs, to request a new review or discuss an existing review. To do this, select the “Reader groups” to include just the ACs, SACs, or PCs, but not reviewers.

<!-- what else should we mention in the message? -->
<!-- should we mention issues with tone of review? aggressively confused. -->
<!-- - explain our efforts to engage with the reviewer in our rebuttal: i.e., we explain terminology (and that it is standard), we give a review of the field, we explain our proposed method again in simple terms, and explain its significance.
 -->
<!-- We made an effort to engage with this reviewer through our rebuttal, including attempting to provide an overview of the literature to help them provide a competent evaluation. -->



---
<!-- possible message about wwyA if they don't respond to us -->
<!-- - Reviewer `wwyA` has misunderstood our proposed method. They claim that it is a "convolution network [meaning 2D convolution in standard CNNs] applied to the attention scores". This justifies weakness 1: "novelty is limited". This is an incorrect characterization our proposed method. Our proposal involves defining a novel kind of convolution operation which is entirely distinct from  2D spatial convolutions, despite the resemblance in the name. Moreover, likely due to this misunderstanding, the reviewer claims our algorithm is computationally inefficient with a complexity of $O(n^3)$. This is in fact incorrect. The computational complexity of each step is described in the paper and the overall complexity is linear in $n$. We explain both these points in our rebuttal. -->

## Follow up message: Bad faith behavior from reviewer xLCw

Dear AC,

We would like to follow up on our earlier message about reviewer xLCw. Earlier, we raised our concern about this review being uninformative and unprofessional. In particular, the review did not discuss the content of our paper. Rather, the review focused entirely on confusion around terminology which are standard in this line of work. We point out that no other review had this issue.

We wrote a rebuttal which aimed to engage with the reviewer to clarify their confusion and provide them with the needed background to understand our work. Our rebuttal was factual, constructive, and understanding. The following is an excerpt from the opening of our rebuttal.

> We understand that reviewers have varying backgrounds and levels of expertise in different areas of research. We also understand that if the reviewer did not grasp the basic setting they would be unable to evaluate the technical content. We will aim to constructively engage with the reviewer to help them understand this line of work better. To address these issues, our rebuttal will be structured as follows:
>
> 1. Clarification on the terminology used in this line of work
> 2. An introductory overview of existing methods in this literature
> 3. A summary and reminder of the proposed methodology and contributions of this paper

In our rebuttal, we go on to give a thorough explanation of both our work as well as the related literature in hopes of helping the reviewer understand and provide an informative review.

Our rebuttal was understanding,
> We understand that reviewers have varying backgrounds and levels of expertise in different areas of research. 

constructive,
> We will aim to constructively engage with the reviewer to help them understand this line of work better. To address these issues, our rebuttal will be structured as follows: [...]

polite,
> Does this clarify your confusion about terminology and the setting? Please let us know if you have any further questions and we would be happy to clarify further.

and inviting,
> Please let us know if you have any further questions or comments. We look forward to hearing back from you.

The reviewer responded to our rebuttal in a defensive and unprofessional manner. Their response is below.

> There is not need to be rude in you response.
> 
> The paper needs to be understandable as a stand-alone paper for a general conference such as ICML. You need to define the terms you are using, particularly when you are using words with standard meanings with technical terms that do not match the normal meaning. Is there no definition you can provide? I think I will downgrade my rating.

The reviewer has refused to engage in the review process in good faith and has instead turned to personal attacks and *threatened to downgrade their rating in retaliation*. We note that this reviewer had an unprofessional tone in their initial review as well. We believe this is inappropriate behavior for any venue, and certainly violates ICML's code of conduct. We are also concerned about the effect this will have on the other reviewers.

We kindly request that this review be removed. We would appreciate your advice on how to proceed given this code of conduct violation. What is the procedure for raising such issues and how does ICML handle such things?

We would appreciate your response on this matter.


Thank you sincerely,

Authors


<!-- outline: main points to cover -->
- The original review did not mention anything about the content of the paper. It only expressed confusion around terminology.
- The reviewer's comments in the original review had an unprofessional and confrontational tone:
> you should say so up-front
> you seem to be couching it in terms of...
- Our rebuttal aimed to engage with the reviewer to help them understand our paper and the line of work it is a part of to help them provide an informative review. Our rebuttal addressed the content of the review factually and constructively.
- The reviewer responded in bad faith. Their response contains personal attacks and is threatening to downgrade their rating. This constitutes "retaliation", which is against the ICML code of conduct.