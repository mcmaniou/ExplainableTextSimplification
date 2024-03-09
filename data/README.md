## Datasets

Initial experimentation was performed using the following two datasets:
1. **CELLS** dataset:includes sets of abstracts and Lay Language Summaries (LLS) of the 62886 biomedical articles.

    *Y. Guo, W. Qiu, G. Leroy, S. Wang, and T. Cohen. Cells: A parallel corpus for biomedical lay language generation. arXiv, 2022. doi: https://doi.org/10.48550/arXiv.2211.03818.*

2. Plain Language Adaptation of Biomedical Abstracts (**PLABA**): consists of 749 scientific abstracts and their corresponding adaptations in plain language.

    *K. Attal, B. Ondov, and D. Demner-Fushman. A dataset for plain language adaptation of biomedical abstracts. Scientific Data, 10(1):8, Jan. 2023. ISSN 2052-4463. doi: 10.1038/s41597-022-01920-3. URL https://doi.org/10.1038/s41597-022-01920-3.*

**Only the PLABA dataset was used for the rest of the experiments of the project.** 

The data was analyzed both in per document (abstracts and lay summaries) and per sentence. The initial data are available in the [NATURE_PLABA](https://github.com/mcmaniou/ExplainableTextSimplification/tree/main/data/NATURE_PLABA) folder. And the processed data that will be used for our experiments in the `nature_paragraph_data.csv` and `nature_sentences_data.csv` files.

PLABA dataset information:
| # Complex Sentences  | # Plain Sentences  | # Total Sentences  | # Complex Documents  | # Plain Documents  | # Total Documents  |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 7605 | 8886 | 16492 | 749 | 920  | 1669  |



