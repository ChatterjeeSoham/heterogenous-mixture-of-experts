import pickle

from experiments.experiment_runner import run_experiment

with open("Real_Datasets/gym_input.np","rb") as f:
    X = pickle.load(f)

with open("Real_Datasets/gym_target.np","rb") as f:
    y = pickle.load(f)

run_experiment(
    X,
    y,
    dataset_name="Gym"
)