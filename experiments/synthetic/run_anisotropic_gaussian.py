from Synthetic_Datasets.Anisotropic_Gaussian import generate_Anisotropic_Gaussian
from experiments.experiment_runner import run_experiment

X, y = generate_Anisotropic_Gaussian()

run_experiment(
    X,
    y,
    dataset_name="Anisotropic Gaussian"
)