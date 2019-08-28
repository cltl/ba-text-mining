Offenseval code for NB/SVM
===========================

The Offenseval code is organized in three folders

- tasks: contains classes to extract data related to specific tasks. You can use the model of 'vua_format' to extract data from other tasks/data sets
- ml_pipeline: code for ML pipeline
    * preprocessing: tokenize, lowercase, etc.
    * representation: format data for input to classifiers. Currently allows for count vectors and word embeddings
    * utils: utility functions for data splitting and grid search
    * pipelines: defines pipelines with a given preprocessing, representation and classification step. Current classifiers are Naive Bayes and SVM
    * experiment: contains the main method to run pipelines on a given task
- tests: contains a basic test suite (to be run with pytest), showing usage examples 


Requirements
============

You can create a Python (3) environment with pip or conda, and load required packages as follows.

With venv and pip
-------------------
 
```
   $ python -m venv <env>
   $ source activate <env>
   $ pip install -r requirements.txt
```

With conda
----------

```
$ conda create --name <env> --file requirements.txt
```

Usage
=======

* Make sure that you load or link data files to a `data/` folder under `pynlp/`. 

   * This includes word embeddings. The code currently runs with Glove twitter embeddings or wiki-news embeddings.

* You can use pytest to run the test suite

   * from PyCharm: edit a pytest configuration to run `py.test under test_suite.py`

* The main function is located under `ml_pipeline/experiment.py`.  

   * the function takes three arguments:
       * a task name (default is 'offenseval') specifying data file names
       * the path to the data (default is 'data')
       * a pipeline (default is 'naive_bayes')   

Authors: Sophie Arnoult, Pia Sommerauer

