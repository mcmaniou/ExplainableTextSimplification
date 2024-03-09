# ExplainableTextSimplification
A workflow that improves the readability of complex texts, by performing lexical text simplification. It is built on the following fundamental steps:
1. Classification of an input text.
2. If the text is complex, detection of the complex parts. It employs feature importance interpretability techniques.
3. An algorithm that iteratively performs masking in order to change the complex parts in simpler ones.

The proposed workflow is presented in the following graph:
![Workflow](https://github.com/mcmaniou/ExplainableTextSimplification/blob/main/images/workflow.png "Our workflow")


### Models
The models fine-tuned for this task are BioBERT and DistilBERT. They were trained to receive as input text either a sentence or a paragraph. All the models are available at [this link](https://drive.google.com/drive/folders/1rdDBiC-0jWKRQcippC_GT6P0prhU3Yus?usp=sharing).

As a result, for the classifier we have four models:
| # Name  | # Base Model  | # Class of Training Data  | # Type of Input Data  |
|:---:|:---:|:---:|:---:|
| bert_par_all | BioBERT | all | paragraphs |
| dist_par_all | DistBERT | all | paragraphs |
| bert_sent_all | BioBERT | all | sentences |
| dist_sent_all | DistBERT | all | sentences |

And for the mask-filler we have eight models:
| # Name  | # Base Model  | # Class of Training Data  | # Type of Input Data  |
|:---:|:---:|:---:|:---:|
| bert_par_all | BioBERT | all | paragraphs |
| dist_par_all | DistBERT | all | paragraphs |
| bert_par_plain | BioBERT | plain | paragraphs |
| dist_par_plain | DistBERT | plain | paragraphs |
| bert_sent_all | BioBERT | all | sentences |
| dist_sent_all | DistBERT | all | sentences |
| bert_sent_plain | BioBERT | plain | sentences |
| dist_sent_plain | DistBERT | plain | sentences |

For the masking experiments, we also tried the base models, not at all trained on this data. 


### Project Structure
The project has the following structure:
- code: includes multiple scripts created for the different parts of the project, from data processing to fine-tuning models and evaluation of results.
- data: the datasets used for this project.



