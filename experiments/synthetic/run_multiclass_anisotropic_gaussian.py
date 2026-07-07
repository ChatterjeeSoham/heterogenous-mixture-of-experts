from Synthetic_Datasets.Multiclass_Anisotropic_Gaussian import generate_multiclass_anisotropic_gaussians
from experiments.experiment_runner import run_experiment

X, y = generate_multiclass_anisotropic_gaussians()

run_experiment(
    X,
    y,
    dataset_name="Multiclass Anisotropic Gaussian"
)