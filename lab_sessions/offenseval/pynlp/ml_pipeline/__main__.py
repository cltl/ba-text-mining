from ml_pipeline import experiment
import argparse

parser = argparse.ArgumentParser(description='run classifier on Offenseval data')
parser.add_argument('--task', dest='task', default="offenseval")
parser.add_argument('--data_dir', dest='data_dir', default="data/")
parser.add_argument('--pipeline', dest='pipeline', default='naive_bayes_counts')
args = parser.parse_args()

experiment.run(args.task, args.data_dir, args.pipeline)
