import pandas as pd


class Offenseval:
    """
    Offenseval data from Zampieri et al. (2017)
    three-level classification of tweets:
      1. offensive (OFF) vs not-offensive (NOT)
      2. [OFF only] targeted (TIN) vs non-targeted (UNT)o
      3. [TIN only] targeted to: individuals (IND), groups (GRP) or other (OTH)
    """

    def __init__(self, train_file, test_file, task='subtask_a'):
        self.train_data = pd.read_csv(train_file, delimiter="\t")
        self.test_data = pd.read_csv(test_file,
                                     delimiter="\t",
                                     header=None,
                                     names=["tweet", "subtask_a", "subtask_b", "subtask_c"])
        self.task = task
        self.name = "Offenseval2017"
        self.hapaxes = []

    def __str__(self):
        return self.name + ", " + self.task

    def train_dev_instances(self, proportion_train=0.8, proportion_dev=0.2):
        """returns training and development instances and classes for a given task

        randomly splits training data in training/dev data

        :param proportion_train: proportion of training data to extract as training data
        :param proportion_dev: proportion of training data to extract as dev data
        :param task: classification task, default: "subtask_a"
        :return: X_train, y_train, X_dev, y_dev
        """
        if proportion_train + proportion_dev > 1:
            raise ValueError("proportions of training and dev data may not go beyond 1")
        nb_total = int(round(self.train_data.shape[0] * (proportion_train + proportion_dev)))
        nb_train = int(round(self.train_data.shape[0] * proportion_train))
        df = self.train_data.sample(n=nb_total, random_state=1)
        df_train = df[:nb_train]
        df_dev = df[nb_train:]
        return df_train['tweet'], df_train[self.task], df_dev['tweet'], df_dev[self.task]

    def test_instances(self):
        return self.test_data['tweet'], self.test_data[self.task]
