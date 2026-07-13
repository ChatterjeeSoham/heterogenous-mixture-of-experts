import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from modt.modt import MoDT
from modt._initialization import Random_init

import numpy as np
import matplotlib.pyplot as plt
def generate_piecewise_linear_kink(n=3000, noise=0.35, seed=42):

    np.random.seed(seed)

    X = np.random.uniform(-4, 4, size=(n, 2))
    x, y = X[:, 0], X[:, 1]

    # piecewise-linear boundary parameters
    a1, b1 = 0.7, -0.5
    a2, b2 = -0.6, 0.8

    y_true = np.zeros(n, dtype=int)

    left = x < 0
    right = ~left

    y_true[left] = (y[left] > a1 * x[left] + b1).astype(int)
    y_true[right] = (y[right] > a2 * x[right] + b2).astype(int)

    # add feature noise
    X += noise * np.random.randn(*X.shape)

    return X, y_true


# =========================================================
# Generate Dataset
# =========================================================
X, y = generate_piecewise_linear_kink(n=3000, noise=0.35)



# Generate data
X, y = generate_piecewise_linear_kink(n=3000, noise=0.35)

plt.figure(figsize=(7,7))

plt.scatter(
    X[:,0],
    X[:,1],
    c=y,
    cmap="coolwarm",
    s=10,
    alpha=0.6
)

# ==========================================
# Bayes Optimal Boundary
# ==========================================

x_left = np.linspace(-4, 0, 500)
y_left = 0.7 * x_left - 0.5

x_right = np.linspace(0, 4, 500)
y_right = -0.6 * x_right + 0.8

plt.plot(
    x_left,
    y_left,
    'k',
    linewidth=3,
    label="Bayes Boundary"
)

plt.plot(
    x_right,
    y_right,
    'k',
    linewidth=3
)

# mark kink point
plt.scatter(
    [0],
    [0.8],
    color='black',
    s=50,
    zorder=10
)

plt.title("Piecewise Linear Kink Dataset with Bayes Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.legend()
plt.grid(True)

plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()


# MoDT Parameters (ORIGINAL)
# =========================================================
# Train/Test Split
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=100,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)


# =========================================================

parameters = {
"X": X_train,
"y": y_train,
"n_experts": 3,
"iterations": 50,
"max_depth": 1,
"init_learning_rate": 20,
"learning_rate_decay": 0.99,
"initialization_method": Random_init(48),
"use_2_dim_gate_based_on": None,
"use_2_dim_clustering": False,
"black_box_algorithm": None,
"feature_names": None,
"class_names": None,
"save_likelihood": False,
"verbose": True,
}
parameters_fit = {
"optimization_method": "least_squares_linear_regression",
"early_stopping": "accuracy",
"use_posterior": False,
}

# =========================================================

# Train Original MoDT

# =========================================================

modt = MoDT(**parameters)
modt.fit(**parameters_fit)

# =========================================================
# Evaluate
# =========================================================
print("Training accuracy:", modt.score_internal_disjoint())
test_acc = modt.score(X_test, y_test)
print("Test accuracy:", test_acc)

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Grid
# ==========================================

x_min, x_max = X_train[:,0].min()-1, X_train[:,0].max()+1
y_min, y_max = X_train[:,1].min()-1, X_train[:,1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# ==========================================
# MoDT prediction
# ==========================================

Z = modt.predict(grid)

Z = np.asarray(Z).reshape(xx.shape)

# ==========================================
# Plot
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(
    X_train[:,0],
    X_train[:,1],
    c=y_train,
    cmap="coolwarm",
    s=8,
    alpha=0.5
)

plt.contour(
    xx,
    yy,
    Z,
    levels=[0.5],
    colors="black",
    linewidths=3
)

plt.title("Homogeneous MoDT Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.gca().set_aspect("equal")
plt.tight_layout()

plt.show()

# =========================================================
# Weighted QDA
# =========================================================
class WeightedQDA:

    def __init__(self, reg_param=1e-2):
        self.reg_param = reg_param

    def fit(self, X, y, sample_weight):

        X = np.asarray(X)
        y = np.asarray(y)
        w = np.asarray(sample_weight)

        self.classes_ = np.unique(y)
        d = X.shape[1]

        self.means_ = {}
        self.covs_ = {}
        self.priors_ = {}

        for c in self.classes_:

            mask = (y == c)

            Xc = X[mask]
            wc = w[mask]

            wc_sum = wc.sum()
            if wc_sum == 0:
                wc_sum = 1e-8

            mu = np.average(Xc, axis=0, weights=wc)

            X_centered = Xc - mu

            cov = (X_centered.T * wc) @ X_centered / wc_sum

            cov += self.reg_param * np.eye(d)

            self.means_[c] = mu
            self.covs_[c] = cov
            self.priors_[c] = wc_sum / w.sum()

        return self

    def _log_gaussian(self, X, mean, cov):

        Xc = X - mean

        inv_cov = np.linalg.inv(cov)

        log_det = np.linalg.slogdet(cov)[1]

        return -0.5 * (np.sum(Xc @ inv_cov * Xc, axis=1) + log_det)

    def predict_proba(self, X):

        X = np.asarray(X)

        log_probs = []

        for c in self.classes_:

            log_prior = np.log(self.priors_[c] + 1e-12)

            log_lik = self._log_gaussian(X, self.means_[c], self.covs_[c])

            log_probs.append(log_prior + log_lik)

        log_probs = np.vstack(log_probs).T

        log_probs -= log_probs.max(axis=1, keepdims=True)

        probs = np.exp(log_probs)

        probs /= probs.sum(axis=1, keepdims=True)

        return probs

    def predict(self, X):

        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# =========================================================
# Identifiable Heterogeneous MoDT
# =========================================================
class HeteroMoDT_Identifiable(MoDT):

    def _update_gating(self):

        super()._update_gating()

        shift = self.gating_weights[:, -1].copy()

        self.gating_weights -= shift[:, None]

    def _train_trees(self):

        self.DT_experts = []

        experts = [

            DecisionTreeClassifier(max_depth=self.max_depth),

            CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=5000, dual=False),
                method="sigmoid"
            ),

            WeightedQDA(reg_param=1e-2)
        ]

        for j, model in enumerate(experts):

            model.fit(
                self.X,
                self.y,
                sample_weight=self.gating_values[:, j]
            )

            self.DT_experts.append(model)


# =========================================================
# Train MoDT
# =========================================================
hetero_modt = HeteroMoDT_Identifiable(

    X=X_train,
    y=y_train,

    n_experts=3,

    iterations=30,

    max_depth=3,

    init_learning_rate=20,

    learning_rate_decay=0.99,

    initialization_method=Random_init(40),

    use_2_dim_gate_based_on=None,

    verbose=False
)

hetero_modt.fit(
    optimization_method="least_squares_linear_regression",
    early_stopping=False
)


# =========================================================
# Evaluate
# =========================================================
hetero_acc = hetero_modt.score(X_test, y_test)

print("\nIdentifiable Hetero-MoDT TEST accuracy:", hetero_acc)


# =========================================================
# Expert Usage
# =========================================================
from collections import Counter

expert_ids = np.argmax(hetero_modt.gating_values, axis=1)

usage = Counter(expert_ids)

total = len(expert_ids)

print("\nExpert usage (training data):")

print(f"Expert 0 (Decision Tree): {usage.get(0,0)} ({usage.get(0,0)/total:.2%})")
print(f"Expert 1 (Linear SVM)   : {usage.get(1,0)} ({usage.get(1,0)/total:.2%})")
print(f"Expert 2 (Weighted QDA): {usage.get(2,0)} ({usage.get(2,0)/total:.2%})")


expert_assignment = np.argmax(
    hetero_modt.gating_values,
    axis=1
)

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    X_train[:,0],
    X_train[:,1],
    c=expert_assignment,
    cmap="Set1",
    s=15
)

plt.colorbar(scatter)

plt.title("Dominant Expert Assignment")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()


# =====================================================
# Create Grid
# =====================================================

x_min, x_max = X_train[:,0].min()-1, X_train[:,0].max()+1
y_min, y_max = X_train[:,1].min()-1, X_train[:,1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# add MoDT bias column
grid_modt = np.column_stack([
    grid,
    np.ones(len(grid))
])

# =====================================================
# Gate Probabilities
# =====================================================

gate_probs = hetero_modt._gating_softmax(
    grid_modt,
    hetero_modt.theta_gating
)

expert_region = np.argmax(gate_probs, axis=1)

# =====================================================
# Expert Predictions
# =====================================================

tree = hetero_modt.DT_experts[0]
svm  = hetero_modt.DT_experts[1]
qda  = hetero_modt.DT_experts[2]

pred_tree = tree.predict(grid_modt)
pred_svm  = svm.predict(grid_modt)
pred_qda  = qda.predict(grid_modt)

# =====================================================
# Piecewise MoE Prediction
# =====================================================

final_pred = np.zeros(len(grid_modt))

final_pred[expert_region == 0] = pred_tree[
    expert_region == 0
]

final_pred[expert_region == 1] = pred_svm[
    expert_region == 1
]

final_pred[expert_region == 2] = pred_qda[
    expert_region == 2
]

# reshape for plotting

final_pred = final_pred.reshape(xx.shape)
expert_region_plot = expert_region.reshape(xx.shape)

# =====================================================
# Plot
# =====================================================

plt.figure(figsize=(10,8))

# final classifier surface
plt.contourf(
    xx,
    yy,
    final_pred,
    levels=[-0.5,0.5,1.5],
    alpha=0.35
)

# expert routing boundaries
plt.contour(
    xx,
    yy,
    expert_region_plot,
    levels=[0.5,1.5],
    colors='black',
    linewidths=2
)

# training points
plt.scatter(
    X_train[:,0],
    X_train[:,1],
    c=y_train,
    cmap='viridis',
    s=8
)

plt.title("Heterogeneous MoDT Decision Surface")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Expert predictions on grid
# =====================================================

tree = hetero_modt.DT_experts[0]
svm  = hetero_modt.DT_experts[1]
qda  = hetero_modt.DT_experts[2]

Z_tree = tree.predict(grid_modt).reshape(xx.shape)
Z_svm  = svm.predict(grid_modt).reshape(xx.shape)
Z_qda  = qda.predict(grid_modt).reshape(xx.shape)

# =====================================================
# Expert regions from gating
# =====================================================

expert_map = expert_region.reshape(xx.shape)

tree_region = (expert_map == 0)
svm_region  = (expert_map == 1)
qda_region  = (expert_map == 2)

# =====================================================
# Mask each expert outside its region
# =====================================================

Z_tree_masked = np.where(tree_region, Z_tree, np.nan)
Z_svm_masked  = np.where(svm_region,  Z_svm,  np.nan)
Z_qda_masked  = np.where(qda_region,  Z_qda,  np.nan)

# =====================================================
# Plot
# =====================================================

plt.figure(figsize=(10,8))

# lightly show expert ownership
plt.contourf(
    xx,
    yy,
    expert_map,
    levels=np.arange(4)-0.5,
    alpha=0.15,
    cmap="Set1"
)

# Tree boundary only in Tree region
plt.contour(
    xx,
    yy,
    Z_tree_masked,
    levels=[0.5],
    linewidths=3,
    linestyles='solid'
)

# SVM boundary only in SVM region
plt.contour(
    xx,
    yy,
    Z_svm_masked,
    levels=[0.5],
    linewidths=3,
    linestyles='dashed'
)

# QDA boundary only in QDA region
plt.contour(
    xx,
    yy,
    Z_qda_masked,
    levels=[0.5],
    linewidths=3,
    linestyles='dashdot'
)

# training data
plt.scatter(
    X_train[:,0],
    X_train[:,1],
    c=y_train,
    cmap="coolwarm",
    s=8
)

plt.title("Heterogeneous MoDT: Expert Boundaries in Their Own Regions")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()

plt.show()

