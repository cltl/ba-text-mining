import argparse
import logging
import sys

from tasks import offenseval as of
from tasks import vua_format as vf
from ml_pipeline import utils
from ml_pipeline import pipelines


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def run(task_name, data_dir, pipeline_name):
    logger.info('>> Running {} experiment'.format(task_name))
    tsk = task(task_name)
    logger.info('>> Loading data...')
    tsk.load(data_dir)
    logger.info('>> retrieving train/test instances...')
    train_X, train_y, test_X, test_y = utils.get_instances(tsk, split_train_dev=False)
    logger.info('>> training pipeline ' + pipeline_name)
    pipe = pipeline(pipeline_name)
    pipe.fit(train_X, train_y)
    logger.info('>> testing...')
    sys_y = pipe.predict(test_X)
    logger.info('>> evaluation...')
    logger.info(utils.eval(test_y, sys_y))


def task(name):
    if name == 'offenseval':
        return of.Offenseval()
    elif name == 'vua-format':
        return vf.VuaFormat()
    else:
        raise ValueError("task name is unknown. You can add a custom task in 'tasks'")


def pipeline(name):
    if name == 'naive_bayes':
        return pipelines.naive_bayes()
    elif name == 'svm_libsvc_counts':
        return pipelines.svm_libsvc_counts()
    elif name == 'svm_libsvc_tfidf':
        return pipelines.svm_libsvc_tfidf()
    elif name == 'svm_libsvc_embed':
        return pipelines.svm_libsvc_embed()
    elif name == 'svm_sigmoid_embed':
        return pipelines.svm_sigmoid_embed()
    else:
        raise ValueError("pipeline name is unknown. You can add a custom pipeline in 'pipelines'")


#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description='run classifier on Offenseval data')
#    parser.add_argument('--task', dest='task', default="offenseval")
#    parser.add_argument('--data_dir', dest='data_dir', default="../data/")
#    parser.add_argument('--pipeline', dest='pipeline', default='naive_bayes')
#    args = parser.parse_args()

#    run(args.task, args.data_dir, args.pipeline)

