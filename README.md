# Heterogeneous Mixture of Experts for Interpretable Machine Learning

Official implementation of the paper

**Heterogeneous Mixture of Experts for Interpretable Machine Learning**

**Authors**

- Smarajit Bose
- Soham Chatterjee
- Rwitobroto Dey

Indian Statistical Institute, Kolkata, India

---

## Overview

This repository contains the official implementation of the proposed **Heterogeneous Mixture of Experts (Hetero-Mix)** framework for interpretable machine learning.

Unlike conventional Mixture-of-Experts models that employ a single expert family, the proposed framework combines heterogeneous experts under a unified probabilistic gating mechanism.

The current implementation uses

- Decision Trees
- Linear Support Vector Machines
- Quadratic Discriminant Analysis

as expert models within a common softmax gating architecture.

The optimization is performed using a generalized Expectation-Maximization (GEM) algorithm together with a regression-based gating update.

---

## Repository Structure

```
heterogeneous-mixture-of-experts/

│
├── modt/                         # Original MoDT implementation
│
├── src/                          # Proposed Heterogeneous Mixture implementation
│
├── Synthetic_Datasets/           # Synthetic dataset generators
│
├── Real_Datasets/                # Real benchmark datasets
│
├── experiments/
│   ├── experiment_runner.py      # Common experiment pipeline
│   ├── synthetic/                # Synthetic dataset experiments
│   ├── real/                     # Real dataset experiments
│   └── single_expert/            # Single-expert comparison experiments
│
├── figures/
│   ├── scripts/                  # Figure generation scripts
│   └── plots/                    # Generated figures
│
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/ChatterjeeSoham/heterogeneous-mixture-of-experts.git

cd heterogeneous-mixture-of-experts
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running Experiments

### Synthetic datasets

Run the corresponding script inside

```
experiments/synthetic/
```

Example

```bash
python piecewise_linear_kink.py
```

---

### Real datasets

Run the corresponding script inside

```
experiments/real/
```

Example

```bash
python adult.py
```

---

### Common Experiment Pipeline

Most experiments use

```
experiments/experiment_runner.py
```

which performs

- 10 random train-test splits
- Standardization
- Training of Random Forest
- Training of MoDT
- Training of Heterogeneous Mixture
- Expert utilization analysis
- Mean ± Standard Deviation reporting

---

## Implemented Experts

- Decision Tree
- Linear SVM (probability calibrated)
- Weighted QDA

---

## Experimental Evaluation

The implementation reproduces the experiments reported in the paper, including

- Real-world benchmark datasets
- Synthetic datasets
- Expert utilization analysis
- Decision boundary visualizations

---

## Citation

If you use this repository, please cite the accompanying paper.

```
Citation will be added after publication.
```

---

## License

MIT License
