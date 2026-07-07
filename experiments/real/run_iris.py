import pandas as pd

from experiments.experiment_runner import run_experiment

X = pd.read_pickle("Real_Datasets/iris_input.pd")
y = pd.read_pickle("Real_Datasets/iris_target.pd")

run_experiment(
    X,
    y,
    dataset_name="Iris"
)