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
Environment
------------
You can create a Python (3) environment (for instance 'myenv') with pip or conda, and load required packages as follows.

With venv and pip
"""""""""""""""""
 
```
   $ python -m venv myenv
   $ source activate myenv
```

With conda
"""""""""""""""""

```
   $ conda create --name myenv python=3.6
   $ conda activate myenv
```

Requirements
-------------
Install the following libraries, as well as the spacy model 'en_core_web_sm':

```
   $ pip install Keras scikit-learn spacy nltk
   $ python -m spacy download en_core_web_sm
```

Usage
=======

* Make sure that you load or link data files to a `data/` folder under `pynlp/`. 

   * This includes word embeddings. The code currently runs with Glove twitter embeddings or wiki-news embeddings.

* You can use pytest to run the test suite

   * from PyCharm: edit a pytest configuration to run `py.test under test_suite.py`. Use the absolute path to 'pynlp' as working directory.
   * from the command line (under 'pynlp'): call 'pytest'; this will run all test suites under 'tests'

* Call the 'ml_pipeline' module to run experiments:

   * ```pynlp$ python -m ml_pipeline```  
   * from PyCharm: edit a run configuration, setting the working directory to 'pynlp'
   * the main function takes three arguments:
       * a task name (default is 'offenseval') specifying data file names
       * the path to the data (default is 'data')
       * a pipeline (default is 'naive_bayes')   

Authors: Sophie Arnoult, Pia Sommerauer

