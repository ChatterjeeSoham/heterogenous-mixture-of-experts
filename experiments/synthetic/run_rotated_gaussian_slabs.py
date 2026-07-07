from Synthetic_Datasets.Rotated_gaussian_slabs import generate_Rotated_Gaussian_Slabs
from experiments.experiment_runner import run_experiment

X, y = generate_Rotated_Gaussian_Slabs()

run_experiment(
    X,
    y,
    dataset_name="Rotated Gaussian Slabs"
)