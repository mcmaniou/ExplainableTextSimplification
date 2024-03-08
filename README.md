# ExplainableTextSimplification
A workflow that improves the readability of complex texts, by performing lexical text simplification. It is built on the following fundamental steps:
1. Classification of an input text.
2. If the text is complex, detection of the complex parts. It employs feature importance interpretability techniques.
3. An algorithm that iteratively performs masking in order to change the complex parts in simpler ones.

The project has the following structure:
- code: includes multiple scripts created for the different parts of the project, from data processing to fine-tuning models and evaluation of results.
- data: the datasets used for this project
- results: the results of workflow

