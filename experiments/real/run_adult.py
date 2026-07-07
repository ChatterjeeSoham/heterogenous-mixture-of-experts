import pickle
import numpy as np
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from modt.modt import MoDT
from modt._initialization import Random_init

from src.hetero_mix import HeteroMix


# =====================================================
# LOAD DATA
# =====================================================

with open("Real_Datasets/adult_train_input.np", "rb") as f:
    X_train = pickle.load(f)

with open("Real_Datasets/adult_train_target.np", "rb") as f:
    y_train = pickle.load(f)

with open("Real_Datasets/adult_test_input.np", "rb") as f:
    X_test = pickle.load(f)

with open("Real_Datasets/adult_test_target.np", "rb") as f:
    y_test = pickle.load(f)

def run_experiment_adult(
        X_train,
        y_train,
        X_test,
        y_test,
        dataset_name="Adult"):

    DEPTHS = [2, 3]
    N_RUNS = 10

    print("=" * 70)
    print(f"Dataset : {dataset_name}")
    print("=" * 70)

    for depth in DEPTHS:

        print("\n" + "=" * 60)
        print(f"Running experiments for max_depth = {depth}")
        print("=" * 60)

        modt_test_accuracies = []
        hetero_test_accuracies = []
        rf_test_accuracies = []

        tree_usages = []
        svm_usages = []
        qda_usages = []

        for run in range(N_RUNS):

            print(f"Run {run+1}/{N_RUNS}")

            # -------------------------------------------------
            # Standardization
            # -------------------------------------------------

            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # -------------------------------------------------
            # Random Forest
            # -------------------------------------------------

            rf = RandomForestClassifier(
                n_estimators=200,
                random_state=40
            )

            rf.fit(X_train_scaled, y_train)

            rf_test_accuracies.append(
                rf.score(X_test_scaled, y_test)
            )

            # -------------------------------------------------
            # Original MoDT
            # -------------------------------------------------

            parameters = {

                "X": X_train_scaled,
                "y": y_train,

                "n_experts": 3,

                "iterations": 50,

                "max_depth": depth,

                "init_learning_rate": 20,

                "learning_rate_decay": 0.99,

                "initialization_method": Random_init(40),

                "use_2_dim_gate_based_on": None,

                "use_2_dim_clustering": False,

                "black_box_algorithm": None,

                "feature_names": None,

                "class_names": None,

                "save_likelihood": False,

                "verbose": False
            }

            parameters_fit = {

                "optimization_method":
                "least_squares_linear_regression",

                "early_stopping": "accuracy",

                "use_posterior": False
            }

            modt = MoDT(**parameters)

            modt.fit(**parameters_fit)

            modt_test_accuracies.append(
                modt.score(X_test_scaled, y_test)
            )

            # -------------------------------------------------
            # Heterogeneous Model
            # -------------------------------------------------

            hetero_model = HeteroMix(

                X=X_train_scaled,
                y=y_train,

                n_experts=3,

                iterations=50,

                max_depth=depth,

                init_learning_rate=20,

                learning_rate_decay=0.99,

                initialization_method=Random_init(40),

                use_2_dim_gate_based_on=None,

                verbose=False
            )

            hetero_model.fit(

                optimization_method=
                "least_squares_linear_regression",

                early_stopping=False
            )

            hetero_test_accuracies.append(
                hetero_model.score(X_test_scaled, y_test)
            )

            # -------------------------------------------------
            # Expert Usage
            # -------------------------------------------------

            expert_ids = np.argmax(
                hetero_model.gating_values,
                axis=1
            )

            usage = Counter(expert_ids)

            total = len(expert_ids)

            tree_usages.append(
                usage.get(0, 0) / total * 100
            )

            svm_usages.append(
                usage.get(1, 0) / total * 100
            )

            qda_usages.append(
                usage.get(2, 0) / total * 100
            )

        print("\nRESULTS")
        print("-" * 50)

        print(
            f"Random Forest : "
            f"{np.mean(rf_test_accuracies):.4f} ± {np.std(rf_test_accuracies):.4f}"
        )

        print(
            f"MoDT          : "
            f"{np.mean(modt_test_accuracies):.4f} ± {np.std(modt_test_accuracies):.4f}"
        )

        print(
            f"HeteroMix     : "
            f"{np.mean(hetero_test_accuracies):.4f} ± {np.std(hetero_test_accuracies):.4f}"
        )

        print()

        print(
            f"Decision Tree Expert : "
            f"{np.mean(tree_usages):.2f}%"
        )

        print(
            f"Linear SVM Expert    : "
            f"{np.mean(svm_usages):.2f}%"
        )

        print(
            f"Weighted QDA Expert  : "
            f"{np.mean(qda_usages):.2f}%"
        )