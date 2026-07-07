from Synthetic_Datasets.Hierarchical_Gaussians import generate_hierarchical_gaussians
from experiments.experiment_runner import run_experiment

X, y = generate_hierarchical_gaussians()

run_experiment(
    X,
    y,
    dataset_name="Hierarchical gaussians"
)