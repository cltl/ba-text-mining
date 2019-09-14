import pandas as pd


class VuaFormat:
    """VUA-format data"""

    def __init__(self):
        self.training_file = 'trainData.csv'
        self.test_file = 'testData.csv'
        self.task = None
        self.name = "VUA_format"
        self.train_data = None
        self.test_data = None
        self.dev_data = []

    def __str__(self):
        return self.name + ", " + self.task


            
    def load(self, data_dir, test_files=[]):
        self.train_data = pd.read_csv(data_dir + self.training_file, delimiter="\t")
        #self.dev_data = pd.read_csv(data_dir + self.dev_file, delimiter="\t", header=None,names=["Id", "Text", "Label"])
        #self.test_data = pd.read_csv(data_dir + self.test_file, delimiter="\t", header=None,names=["Id", "Text", "Label"])
        self.test_data = pd.read_csv(data_dir + self.test_file, delimiter="\t")

        #for f in test_files:
            #self.test_data.append(pd.read_csv(data_dir + f, delimiter="\t"))

    def train_instances(self):
        """returns training instances and labels for a given task

        :return: X_train, y_train
        """
        return self.train_data['Text'], self.train_data['Label']

    def test_instances(self):
        return self.test_data['Text'], self.test_data['Label']
