from nltk import TweetTokenizer
import spacy


class Preprocessor:
    """preprocesses the data with NLTK and Spacy (lemmatizer)
    
    TODO remove stopwords"""
    def __init__(self, tokenize, normalize_tweet, lowercase, lemmatize):
        tt_args = {}
        tt_args['reduce_len'] = normalize_tweet
        tt_args['strip_handles'] = normalize_tweet
        tt_args['preserve_case'] = not lowercase
        self.processors = []
        if tokenize:
            self.processors.append(tokenize_with(tt_args))
        if lemmatize:
            self.processors.append(lemmatize_with_spacy)

    def transform(self, data):
        for p in self.processors:
            data = p(data)
        return data

    def fit_transform(self, data, y=None):
        return self.transform(data)


def tokenize_with(kwargs):
    tokenizer = TweetTokenizer(**kwargs)

    def tweet_tokenizer(data):
        return [' '.join(tokenizer.tokenize(tweet)) for tweet in data]
    return tweet_tokenizer


def lemmatize_with_spacy(data):
    nlp = spacy.load("en_core_web_sm")

    print('ok')
    def apply_spacy(tw):
        return ' '.join([token.lemma_ for token in nlp(tw)])
    return [apply_spacy(tweet) for tweet in data]


# -------------- standard preprocessor --------------------------------

def std_prep():
    return Preprocessor(tokenize=True, normalize_tweet=True, lowercase=True, lemmatize=False)
