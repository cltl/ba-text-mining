Offenseval code for NB/SVM
===========================

The Offenseval code is organized in three folders

- data_extraction: 
    * extracts data for a given offenseval task (a, b or c)
    * code for splitting training data in train/dev data
    * preprocessing code for tweets with TweetTokenizer
- classification:
    * Tweet representation:
        * Vocabulary count vectors
        * Embedding representations (added by Pia)
    * defines classifiers as part of a pipeline consisting of preprocessing, data representation and classifier proper
    * grid-search function
    * defines NB and SVM classifiers with decent settings
- [MAIN] tasks: 
    * defines a ML Suite coupling a data extractor with a list of classifiers for experiments on dev/test data
    * records some grid search settings 
    * calls offenseval task to either perform grid search or compare classifiers in a same run


Created by Sophie Arnoult (with modifications by Pia Sommerauer)
