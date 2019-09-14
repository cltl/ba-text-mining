from tasks import offenseval as of
from tasks import vua_format as vf
from ml_pipeline import utils
from ml_pipeline import preprocessing
from ml_pipeline import representation
from ml_pipeline import pipelines


offenseval_data_dir = 'data/'
trac_data_dir = 'resources/TRAC2018/VUA_format/'
hate_speech_data_dir = 'resources/hate-speech-dataset-vicom/VUA_format/'


def test_offenseval_data_extraction_task_a():
    task = of.Offenseval()
    task.load(offenseval_data_dir)
    train_X, train_y = task.train_instances()
    test_X, test_y = task.test_instances()
    assert len(train_X) == 13240
    assert len(test_y) == 319
    assert isinstance(train_X[0], str)

    labels = set(test_y)
    assert len(labels) == 2


def test_data_split_on_offenseval():
    task = of.Offenseval()
    task.load(offenseval_data_dir)
    train_X, train_y, test_X, test_y = utils.get_instances(task, split_train_dev=False)
    assert len(train_X) == 13240
    assert len(test_X) == 319
    assert isinstance(train_X[0], str)

    train_X, train_y, test_X, test_y = utils.get_instances(task, split_train_dev=True)
    assert len(train_X) == 13240 * 0.9
    assert len(test_X) == 13240 * 0.1


def test_preprocessors():
    task = of.Offenseval()
    task.load(offenseval_data_dir)
    train_X, train_y, test_X, test_y = utils.get_instances(task, split_train_dev=True, proportion_train=0.1,
                                                           proportion_dev=0.01)

    prep = preprocessing.Preprocessor(tokenize=False, normalize_tweet=False, lowercase=False, lemmatize=False)
    train_X_prep = prep.transform(train_X)
    assert len(train_X_prep) == len(train_X)
    assert isinstance(train_X_prep[0], str)

    prep = preprocessing.Preprocessor(tokenize=True, normalize_tweet=True, lowercase=True, lemmatize=True)
    train_X_prep = prep.transform(train_X)
    assert len(train_X_prep) == len(train_X)
    assert isinstance(train_X_prep[0], str)


def test_representation():
    task = of.Offenseval()
    task.load(offenseval_data_dir)
    train_X, train_y, test_X, test_y = utils.get_instances(task, split_train_dev=True, proportion_train=0.1,
                                                           proportion_dev=0.01)
    prep = preprocessing.Preprocessor(tokenize=True, normalize_tweet=False, lowercase=False, lemmatize=False)
    train_X = prep.transform(train_X)

    frmt = representation.count_vectorizer()
    train_X = frmt.fit_transform(train_X, train_y)
    assert not isinstance(train_X[0], str)


def test_naive_bayes_pipeline():
    task = of.Offenseval()
    task.load(offenseval_data_dir)
    train_X, train_y, test_X, test_y = utils.get_instances(task, split_train_dev=True, proportion_train=0.1,
                                                           proportion_dev=0.01)
    pipe = pipelines.naive_bayes_counts()
    pipe.fit(train_X, train_y)
    sys_y = pipe.predict(test_X)
    assert len(sys_y) == len(test_y)


def test_grid_search():
    task = of.Offenseval()
    task.load(offenseval_data_dir)
    train_X, train_y, test_X, test_y = utils.get_instances(task, split_train_dev=True, proportion_train=0.1,
                                                           proportion_dev=0.01)
    params = {'clf__C': (0.1, 1)}
    best_sys_y = utils.grid_search(pipelines.svm_libsvc_counts(), params, train_X, train_y, test_X)
    assert len(best_sys_y) == len(test_y)


def test_trac2018():
    task = vf.VuaFormat()
    task.load(trac_data_dir, 'devData.csv')
    train_X, train_y, test_X, test_y = utils.get_instances(task, split_train_dev=True, proportion_train=0.1,
                                                           proportion_dev=0.01)
    pipe = pipelines.naive_bayes_counts()
    pipe.fit(train_X, train_y)
    sys_y = pipe.predict(test_X)
    assert len(sys_y) == len(test_y)


def test_hate_speech():
    task = vf.VuaFormat()
    task.load(hate_speech_data_dir)
    train_X, train_y, test_X, test_y = utils.get_instances(task, split_train_dev=True, proportion_train=0.1,
                                                           proportion_dev=0.01)
    pipe = pipelines.naive_bayes_counts()
    pipe.fit(train_X, train_y)
    sys_y = pipe.predict(test_X)
    assert len(sys_y) == len(test_y)


