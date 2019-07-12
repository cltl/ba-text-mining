Offenseval code for NB/SVM
===========================

The Offenseval code is organized in three folders

- tasks: contains classes to extract data related to specific tasks. You can use the model of the 'offenseval' to extract data from other tasks/data sets
- ml_pipeline: code for ML pipeline
    * preprocessing: tokenize, lowercase, etc.
    * representation: format data for input to classifiers. Currently allows for count vectors and word embeddings
    * utils: utility functions for data splitting and grid search
    * pipelines: defines pipelines with a given preprocessing, representation and classification step. Current classifiers are Naive Bayes and SVM
    * experiment: contains the main method to run pipelines on a given task
- tests: contains a basic test suite (to be run with pytest), showing usage examples 

Authors: Sophie Arnoult, Pia Sommerauer

