from Synthetic_Datasets.Inner_Blob_Outer_Ring import generate_blob_ring_dataset
from experiments.experiment_runner import run_experiment

X, y = generate_blob_ring_dataset()

run_experiment(
    X,
    y,
    dataset_name="Inner Blob + Outer Ring"
)