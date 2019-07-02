from imblearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm, metrics
import numpy as np

#sklearn.preprocessing.FunctionTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin
from gensim.models import KeyedVectors

from data_extraction.preprocessing import Preprocessor, tokenize_tweets, remove_hapaxes, replace_hapaxes, replace_urls, \
    thousand_most_common


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")




class Text2Embedding(TransformerMixin):

  def fit_transform(self, X, parameters=[]):
      print('transforming data using customized transformer')
      #path = '../../../../Data/dsm/word2vec/GoogleNews-vectors-negative300.bin'
      path = '../../../../Data/dsm/word2vec/movies.bin'
      model = KeyedVectors.load_word2vec_format(path, binary=True)
      n_d = len(model['the'])
      print('dimensions:', n_d)
      data = []
      for tokenized_tweet in X:
          tokens = tokenized_tweet.split(' ')
          tweet_matrix = np.array([model[t] for t in tokens if t in model.vocab])
          if len(tweet_matrix)!= 2:
              average_embedding = np.zeros(n_d)
              data.append(average_embedding)
          else:
              data.append(np.mean(tweet_matrix, axis = 0))
      return np.array(data)
  def transform(self, X):
      return self.fit_transform(X)



class Classifier:
    def __init__(self, preprocessor, formatter, classifier, name):
        self.preprocessor = preprocessor
        self.formatter = formatter
        self.clf = classifier
        self.name = name
    
    def train(self, train_X, train_y):

        train_X = self.preprocessor.transform(train_X)
        train_X = self.formatter.fit_transform(train_X)
        print(train_X.shape)
        print(train_y.shape)
        return self.clf.fit(X=train_X, y=train_y)

    def predict(self, test_X):
        test_X = self.preprocessor.transform(test_X)
        test_X = self.formatter.transform(test_X)
        return self.clf.predict(test_X)

    def pipeline(self):
        return Pipeline([('prep', self.preprocessor),
                         ('frm', self.formatter),
                         ('clf', self.clf)])

    def grid_search(self, parameters, train_X, train_y, test_X):
        pipeline = self.pipeline()
        grid_search = GridSearchCV(pipeline, parameters, verbose=1)
        grid_search.fit(train_X, train_y)

        print("best params: {}".format(grid_search.best_params_))
        report(grid_search.cv_results_, n_top=10)
        return grid_search.predict(test_X)


def naive_bayes_classifier(preprocessor=Preprocessor('hapaxes'),
                           formatter=CountVectorizer(),
                           title="Multinomial Naive Bayes"):
    """NB classifier with optimal parameters"""
    return Classifier(preprocessor,
                      formatter,
                      MultinomialNB(),
                      title)


def svm_rbf(preprocessor=Preprocessor(),
            formatter=CountVectorizer(),
            title="SVM"):
    """SVM classifier with rbf kernel"""
    return Classifier(preprocessor,
                      formatter,
                      svm.SVC(gamma='scale'),
                      title)

def svm_rbf_embed(preprocessor=Preprocessor(),
            formatter=Text2Embedding(),
            title="SVM"):
    """SVM classifier with rbf kernel"""
    return Classifier(preprocessor,
                      formatter,
                      svm.SVC(gamma='scale'),
                      title)


def svm_libsvc(preprocessor=Preprocessor(),
               # transform documents (tokens) into a document-word-count vector
               formatter=CountVectorizer(),
               title="SVM with libSVC"):
    return Classifier(preprocessor,
                      formatter,
                      svm.LinearSVC(max_iter=10000, dual=False, C=0.1),
                      title)

def svm_libsvc_embed(preprocessor=Preprocessor(),
               # transform documents (tokens) into a document-word-count vector
               formatter=Text2Embedding(),
               title="SVM with libSVC using embeddings"):
    return Classifier(preprocessor,
                      formatter,
                      svm.LinearSVC(max_iter=10000, dual=False, C=0.1),
                      title)


def svm_sigmoid_embed(preprocessor=Preprocessor(),
               # transform documents (tokens) into a document-word-count vector
               formatter=Text2Embedding(),
               title="SVM with sigmoid using embeddings"):
    return Classifier(preprocessor,
                      formatter,
                      svm.SVC(kernel = 'sigmoid', gamma='scale'),
                      title)
