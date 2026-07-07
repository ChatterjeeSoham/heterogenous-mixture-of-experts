from Synthetic_Datasets.Overlapping_rings import generate_overlapping_rings
from experiments.experiment_runner import run_experiment

X, y = generate_overlapping_rings()

run_experiment(
    X,
    y,
    dataset_name="Overlapping Rings"
)