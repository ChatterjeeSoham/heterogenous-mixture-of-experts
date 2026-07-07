from Synthetic_Datasets.Radial_Sector_Multiclass import generate_radial_sector_dataset
from experiments.experiment_runner import run_experiment

X, y = generate_radial_sector_dataset()

run_experiment(
    X,
    y,
    dataset_name="Radial Sector Dataset"
)