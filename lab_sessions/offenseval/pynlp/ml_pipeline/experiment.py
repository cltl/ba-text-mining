import logging
import sys

from tasks import offenseval as of
from tasks import vua_format as vf
from ml_pipeline import utils, cnn, preprocessing
from ml_pipeline import pipelines
from ml_pipeline.cnn import CNN

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

    if pipeline_name.startswith('cnn'):
        pipe = cnn(pipeline_name)
        train_X, train_y, test_X, test_y = pipe.encode(train_X, train_y, test_X, test_y)
        logger.info('>> testing...')
    else:
        pipe = pipeline(pipeline_name)
  
    logger.info('>> training pipeline ' + pipeline_name)
    pipe.fit(train_X, train_y)

    logger.info('>> testing...')
    sys_y = pipe.predict(test_X)
    logger.info(utils.print_prediction(test_X, test_y, sys_y))
    logger.info('>> evaluation...')
    logger.info(utils.eval(test_y, sys_y))


def task(name):
    if name == 'offenseval':
        return of.Offenseval()
    elif name == 'vua-format':
        return vf.VuaFormat()
    else:
        raise ValueError("task name is unknown. You can add a custom task in 'tasks'")


def cnn(name):
    if name == 'cnn_raw':
        return CNN()
    elif name == 'cnn_prep':
        return CNN(preprocessing.std_prep())
    else:
        raise ValueError("pipeline name is unknown.")


def pipeline(name):
    if name == 'naive_bayes_counts':
        return pipelines.naive_bayes_counts()
    elif name == 'naive_bayes_tfidf':
        return pipelines.naive_bayes_tfidf()
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




