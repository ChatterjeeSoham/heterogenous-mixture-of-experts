import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from Synthetic_Datasets.Hierarchical_Gaussians import (
    generate_hierarchical_gaussians
)

# =====================================================
# LOAD DATA
# =====================================================

X, y = generate_hierarchical_gaussians()

N_RUNS = 10

test_accuracies = []

print("=" * 70)
print("Dataset : Hierarchical Gaussian")
print("Single Expert : QDA")
print("=" * 70)

for run in range(N_RUNS):

    seed = run * 100

    print(f"Run {run+1}/{N_RUNS}")

    # -------------------------------------------------
    # Train / Test Split
    # -------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y
    )

    # -------------------------------------------------
    # Standardization
    # -------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------------------------------
    # QDA
    # -------------------------------------------------

    qda = QuadraticDiscriminantAnalysis(
        reg_param=1e-2
    )

    qda.fit(X_train, y_train)

    test_accuracies.append(
        qda.score(X_test, y_test)
    )

# =====================================================
# FINAL RESULTS
# =====================================================

print("\n================================================")
print("FINAL RESULTS OVER 10 RUNS")
print("================================================")

print(
    f"\nMean Test Accuracy : "
    f"{np.mean(test_accuracies):.4f}"
)

print(
    f"Std Test Accuracy  : "
    f"{np.std(test_accuracies):.4f}"
)