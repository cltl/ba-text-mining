import pandas as pd


class Offenseval:
    """Offenseval data from Zampieri et al. (2017)

    three-level classification of tweets:
      1. offensive (OFF) vs not-offensive (NOT)
      2. [OFF only] targeted (TIN) vs non-targeted (UNT)
      3. [TIN only] targeted to: individuals (IND), groups (GRP) or other (OTH)
    """

    def __init__(self, task='subtask_a'):
        self.training_file = 'offenseval-training-v1.tsv'
        self.test_file = 'offenseval-trial.txt'
        self.task = task
        self.name = "Offenseval2017"
        self.train_data = None
        self.test_data = None

    def __str__(self):
        return self.name + ", " + self.task

    def load(self, data_dir):
        self.train_data = pd.read_csv(data_dir + self.training_file, delimiter="\t")
        self.test_data = pd.read_csv(data_dir + self.test_file,
                                     delimiter="\t",
                                     header=0,
                                     names=["tweet", "subtask_a", "subtask_b", "subtask_c"])

    def train_instances(self):
        """returns training instances and labels for a given task

        :return: X_train, y_train
        """
        return self.train_data['tweet'], self.train_data[self.task]

    def test_instances(self):
        return self.test_data['tweet'], self.test_data[self.task]

