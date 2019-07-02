import re
from collections import Counter
from nltk import TweetTokenizer


def replace_hapaxes_by_UNK(tweet, hapaxes):
    s = []
    for w in tweet.split(' '):
        if w in hapaxes:
            if w[0] == '#':
                s.append('#UNK')
            else:
                s.append('UNK')
        else:
            s.append(w)
    return s


def tokenize_tweets(data):
    tokenizer = TweetTokenizer(reduce_len=True, preserve_case=False, strip_handles=True)
    return [' '.join(tokenizer.tokenize(tweet)) for tweet in data]


def replace_urls(data):
    return [re.sub(r'http\S*', 'URL', tweet) for tweet in data]


def replace_hapaxes(data):
    return handle_hapaxes(data, replace_hapaxes_by_UNK)


def remove_hapaxes_from_tweet(tweet, hapaxes):
    return [w for w in tweet.split(' ') if w not in hapaxes]


def remove_hapaxes(data):
    return handle_hapaxes(data, remove_hapaxes_from_tweet)


def handle_hapaxes(data, method):
    words = []
    for tweet in data:
        words.extend(tweet.split(' '))
    wordcount = Counter(words)
    hapaxes = [w for w, c in wordcount.items() if c == 1]
    return [' '.join(method(tweet, hapaxes)) for tweet in data]


def thousand_most_common(data):
    words = []
    for tweet in data:
        words.extend(tweet.split(' '))
    wordcount = Counter(words)
    common = [w for w, c in wordcount.most_common(1000)]
    return [' '.join([w for w in tweet.split(' ') if w in common]) for tweet in data]


class Preprocessor:
    def __init__(self, filter_tokens=None):
        self.processors = self.append_filter_token_processor([tokenize_tweets], filter_tokens)

    def append_filter_token_processor(self, processors, filter_name):
        if filter_name == 'hapaxes':
            processors.append(remove_hapaxes)
        elif filter_name == '1000_most_common':
            processors.append(thousand_most_common)
        return processors

    def transform(self, data):
        for p in self.processors:
            data = p(data)
        return data

    def fit_transform(self, data, y=None):
        return self.transform(data)

    def set_params(self, **kwargs):
        self.processors = self.append_filter_token_processor([tokenize_tweets], kwargs['filter_tokens'])
