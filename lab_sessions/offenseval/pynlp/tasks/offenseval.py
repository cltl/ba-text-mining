import argparse
import logging
import sys

from sklearn.metrics.classification import classification_report
from sklearn.model_selection import GridSearchCV

from classification.classifier import naive_bayes_classifier as nbc, svm_rbf
from classification.classifier import svm_libsvc, svm_libsvc_embed, svm_rbf_embed, svm_sigmoid_embed
from data_extraction.offenseval2017 import Offenseval
from data_extraction.preprocessing import Preprocessor, tokenize_tweets, remove_hapaxes, replace_hapaxes, replace_urls

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class MLSuite:
    def __init__(self, data_extractor, classifiers, proportion_train=0.9, proportion_dev=0.1, dev_stage=True):
        self.data_extractor = data_extractor
        self.classifiers = classifiers
        self.dev_stage = dev_stage
        if not dev_stage:
            self.proportion_train = 1
            self.proportion_dev = 0
        else:
            self.proportion_train = proportion_train
            self.proportion_dev = proportion_dev


    def extract_train_dev_data(self):
        train_X, train_y, dev_X, dev_y = self.data_extractor.train_dev_instances(self.proportion_train,
                                                                                 self.proportion_dev)
        print('investigating data representation')
        logger.info("   " + str(self.data_extractor))
        logger.info("   Using {} training and {} dev instances".format(len(train_y), len(dev_y)))
        return train_X, train_y, dev_X, dev_y

    def extract_test_data(self):
        test_X, test_y = self.data_extractor.test_instances()
        return test_X, test_y

    def run(self):
        train_X, train_y, dev_X, dev_y = self.extract_train_dev_data()
        if not self.dev_stage:
            test_X, test_y = self.extract_test_data()
        for c in self.classifiers:
            logger.info("   Classifier: {}".format(c.name))
            c.train(train_X, train_y)
            if self.dev_stage:
                logger.info("   Running classifier on dev data")
                pred_y = c.predict(dev_X)
                print('dev labels')
                print(set(train_y))
                logger.info(classification_report(dev_y, pred_y))
            else:
                logger.info("   Running classifier on test data")
                pred_y = c.predict(test_X)
                print('test labels')
                print(set(train_y))
                logger.info(classification_report(test_y, pred_y))


    def grid_search(self, params):
        """performs grid search on first classifier"""
        train_X, train_y, dev_X, dev_y = self.extract_train_dev_data()
        clf = self.classifiers[0]
        pred_y = clf.grid_search(params, train_X, train_y, dev_X)
        logger.info(classification_report(dev_y, pred_y))


# Grid search

def prep_grid_parameters():
    return {'prep__filter_tokens': (None, 'hapaxes', '1000_most_common')}


def svm_clf_grid_parameters():
    return {'clf__class_weight': (None, 'balanced'),
            'clf__dual': (True, False),
            'clf__C': (0.1, 1, 10)}


def svm_rbf_params():
    return {'clf__gamma': (0.01, 0.1, 1),
            'clf__C': (0.01, 0.1, 1)}

#best params: {'clf__C': 1, 'clf__gamma': 0.01}

def grid_search_svm(training_data, test_data):
    mlsuite = MLSuite(Offenseval(training_data, test_data), [svm_rbf_embed()])
    #mlsuite = MLSuite(Offenseval(training_data, test_data), [svm_rbf()])
    parameters = svm_rbf_params()
    mlsuite.grid_search(parameters)


# comparing classifiers

def classifiers():
    return [svm_libsvc_embed()]#svm_libsvc(), nbc(), svm_rbf(), svm_sigmoid_embed()] #, ]


def compare_classifiers(classifiers, training_data, test_data, dev_stage=True):
    mlsuite = MLSuite(Offenseval(training_data, test_data), classifiers, dev_stage=dev_stage)
    logger.info(">> comparing classifiers ({}/{} train/dev split)".format(mlsuite.proportion_train,
                                                                          mlsuite.proportion_dev))
    mlsuite.run()


def run_offenseval_task_a(training_data, test_data):
    """main function

    currently tailored to either
    * perform a grid search
    * or, compare classifiers"""
    #grid_search_svm(training_data, test_data)
    compare_classifiers(classifiers(), training_data, test_data, dev_stage=True)
    #compare_classifiers(classifiers(), training_data, test_data, dev_stage=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run classifier on Offenseval data')
    parser.add_argument('--train', dest='training_data', default="../offenseval2017/offenseval-training-v1.tsv")
    parser.add_argument('--test', dest='test_data', default="../offenseval2017/offenseval-trial.txt")
    args = parser.parse_args()
    run_offenseval_task_a(args.training_data, args.test_data)
