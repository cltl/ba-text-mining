from sklearn.base import TransformerMixin
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


class Text2Embedding(TransformerMixin):

    def fit_transform(self, X, parameters=[]):
        print('transforming data using customized transformer')
        path = '../../../../Data/dsm/word2vec/GoogleNews-vectors-negative300.bin'
        model = KeyedVectors.load_word2vec_format(path, binary=True)
        n_d = len(model['the'])
        print('dimensions:', n_d)
        data = []
        for tokenized_tweet in X:
            tokens = tokenized_tweet.split(' ')
            tweet_matrix = np.array([model[t] for t in tokens if t in model.vocab])
            if len(tweet_matrix) != 2:
                average_embedding = np.zeros(n_d)
                data.append(average_embedding)
            else:
                data.append(np.mean(tweet_matrix, axis=0))
        return np.array(data)

    def transform(self, X):
        return self.fit_transform(X)


# --------------- standard formatters ----------------------

def count_vectorizer(kwargs={}):
    return CountVectorizer(**kwargs)


def text2embeddings():
    return Text2Embedding()
