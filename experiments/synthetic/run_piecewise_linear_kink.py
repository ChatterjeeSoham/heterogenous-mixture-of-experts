from Synthetic_Datasets.Piecewise_Linear_Kink import generate_piecewise_linear_kink
from experiments.experiment_runner import run_experiment

X, y = generate_piecewise_linear_kink()

run_experiment(
    X,
    y,
    dataset_name="Piecewise Linear Kink"
)