import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# =====================================================
# LOAD DATA
# =====================================================

with open("Real_Datasets/gym_input.np", "rb") as f:
    X = pickle.load(f)

with open("Real_Datasets/gym_target.np", "rb") as f:
    y = pickle.load(f)


DEPTHS = [2, 3]
N_RUNS = 10

print("=" * 70)
print("Dataset : Gym")
print("Single Expert : Decision Tree")
print("=" * 70)

for depth in DEPTHS:

    test_accuracies = []

    print("\n" + "=" * 50)
    print(f"Decision Tree (max_depth = {depth})")
    print("=" * 50)

    for run in range(N_RUNS):

        seed = run * 100

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=seed,
            stratify=y
        )

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        dt = DecisionTreeClassifier(
            max_depth=depth,
            random_state=40
        )

        dt.fit(X_train, y_train)

        test_accuracies.append(
            dt.score(X_test, y_test)
        )

    print(
        f"Mean Test Accuracy : "
        f"{np.mean(test_accuracies):.4f}"
    )

    print(
        f"Std Test Accuracy  : "
        f"{np.std(test_accuracies):.4f}"
    )