
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

# =========================================================
# MULTICLASS ANISOTROPIC GAUSSIANS
# =========================================================

def generate_multiclass_anisotropic_gaussians(
    n_per_class=600,
    noise=0.15,
    seed=42
):

    np.random.seed(seed)

    X_list = []
    y_list = []

    # -----------------------------------------------------
    # Covariance orientations
    # -----------------------------------------------------

    angles = [
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4
    ]

    # -----------------------------------------------------
    # Generate each class
    # -----------------------------------------------------

    for k, angle in enumerate(angles):

        # Rotation matrix
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])

        # Anisotropic covariance
        D = np.array([
            [5.0, 0.0],
            [0.0, 0.2]
        ])

        cov = R @ D @ R.T

        Xk = np.random.multivariate_normal(
            mean=[0, 0],
            cov=cov,
            size=n_per_class
        )

        # Additional noise
        Xk += noise * np.random.randn(*Xk.shape)

        yk = np.full(n_per_class, k)

        X_list.append(Xk)
        y_list.append(yk)

    # -----------------------------------------------------
    # Combine classes
    # -----------------------------------------------------

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    return X, y


# =========================================================
# GENERATE DATASET
# =========================================================

X, y = generate_multiclass_anisotropic_gaussians(
    n_per_class=600,
    noise=0.10
)


# =========================================================
# VISUALIZATION
# =========================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap="viridis",
    s=10,
    alpha=0.7
)

plt.title("Multiclass Anisotropic Gaussians")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.grid(True)

plt.gca().set_aspect("equal")

plt.tight_layout()

plt.show()




# =========================================================
# 2. Train/Test Split
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
# 3. Standardization
# =========================================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =========================================================
# 4. Linear SVM Baseline
# =========================================================
svm = CalibratedClassifierCV(
    LinearSVC(C=1.0, max_iter=5000, dual=False),
    method="sigmoid"
)

svm.fit(X_train, y_train)

svm_acc = accuracy_score(y_test, svm.predict(X_test))

print("\nLinear SVM TEST accuracy:", svm_acc)


# =========================================================
# 5. Weighted QDA
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
# 6. Identifiable Heterogeneous MoDT
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
# 7. Train MoDT
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
# 8. Evaluate
# =========================================================
hetero_acc = hetero_modt.score(X_test, y_test)

print("\nIdentifiable Hetero-MoDT TEST accuracy:", hetero_acc)


# =========================================================
# 9. Expert Usage
# =========================================================
from collections import Counter

expert_ids = np.argmax(hetero_modt.gating_values, axis=1)

usage = Counter(expert_ids)

total = len(expert_ids)

print("\nExpert usage (training data):")

print(f"Expert 0 (Decision Tree): {usage.get(0,0)} ({usage.get(0,0)/total:.2%})")
print(f"Expert 1 (Linear SVM)   : {usage.get(1,0)} ({usage.get(1,0)/total:.2%})")
print(f"Expert 2 (Weighted QDA): {usage.get(2,0)} ({usage.get(2,0)/total:.2%})")




import matplotlib.pyplot as plt
import numpy as np

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




x_min, x_max = X_train[:,0].min()-1, X_train[:,0].max()+1
y_min, y_max = X_train[:,1].min()-1, X_train[:,1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

grid = np.c_[xx.ravel(), yy.ravel()]



grid_modt = np.column_stack([
    grid,
    np.ones(len(grid))
])

print(grid_modt.shape)



qda = hetero_modt.DT_experts[2]

Z = qda.predict(grid_modt)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))

plt.contourf(xx, yy, Z, alpha=0.3)

plt.contour(
    xx,
    yy,
    Z,
    levels=[0.5],
    linewidths=2
)

plt.scatter(
    X_train[:,0],
    X_train[:,1],
    c=y_train,
    s=8
)

plt.title("Weighted QDA Boundary")
plt.show()




# =========================================================
# STANDARDIZE
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# FIT QDA
# =========================================================

qda = QuadraticDiscriminantAnalysis()

qda.fit(X_train_scaled, y_train)

# =========================================================
# ACCURACY
# =========================================================

y_pred_train = qda.predict(X_train_scaled)
y_pred_test = qda.predict(X_test_scaled)

print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy :", accuracy_score(y_test, y_pred_test))

# =========================================================
# DECISION BOUNDARY
# =========================================================

h = 0.02

x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

grid = np.c_[xx.ravel(), yy.ravel()]

Z = qda.predict(grid)
Z = Z.reshape(xx.shape)

# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(9, 8))

# decision regions
plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.3,
    cmap="viridis"
)

# boundary lines
plt.contour(
    xx,
    yy,
    Z,
    colors="k",
    linewidths=1
)

# training points
scatter = plt.scatter(
    X_train_scaled[:, 0],
    X_train_scaled[:, 1],
    c=y_train,
    cmap="viridis",
    s=15,
    edgecolor="k",
    alpha=0.8
)

plt.title("QDA Decision Boundary")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.grid(True)

plt.gca().set_aspect("equal")

plt.show()




# =====================================================
# TRUE COVARIANCES USED IN GENERATION
# =====================================================

angles = [
    0,
    np.pi/4,
    np.pi/2,
    3*np.pi/4
]

covs = []

for angle in angles:

    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

    D = np.array([
        [5.0, 0.0],
        [0.0, 0.2]
    ])

    cov = R @ D @ R.T

    covs.append(cov)

covs = np.array(covs)

inv_covs = np.linalg.inv(covs)

# =====================================================
# BAYES PREDICTOR
# =====================================================

def bayes_predict(X):

    scores = []

    for S_inv in inv_covs:

        # x^T Sigma^{-1} x
        q = np.sum((X @ S_inv) * X, axis=1)

        scores.append(q)

    scores = np.column_stack(scores)

    return np.argmin(scores, axis=1)

# =====================================================
# ACCURACY
# =====================================================

y_bayes = bayes_predict(X)

print(
    "Bayes accuracy on generated sample:",
    np.mean(y_bayes == y)
)

# =====================================================
# DECISION BOUNDARY
# =====================================================

h = 0.02

x_min = X[:,0].min() - 1
x_max = X[:,0].max() + 1

y_min = X[:,1].min() - 1
y_max = X[:,1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

grid = np.c_[xx.ravel(), yy.ravel()]

Z = bayes_predict(grid)
Z = Z.reshape(xx.shape)

# =====================================================
# PLOT
# =====================================================

plt.figure(figsize=(8,8))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.30,
    cmap="viridis"
)

plt.contour(
    xx,
    yy,
    Z,
    colors="black",
    linewidths=1
)

plt.scatter(
    X[:,0],
    X[:,1],
    c=y,
    cmap="viridis",
    s=10,
    alpha=0.6
)

plt.title("True Bayes Decision Boundary")
plt.xlabel("x1")
plt.ylabel("x2")

plt.gca().set_aspect("equal")

plt.show()



# =====================================================
# CREATE GRID
# =====================================================

x_min, x_max = X_train[:,0].min()-1, X_train[:,0].max()+1
y_min, y_max = X_train[:,1].min()-1, X_train[:,1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# Add bias column required by the gating network
grid_modt = np.column_stack([
    grid,
    np.ones(len(grid))
])

# =====================================================
# GATING NETWORK
# =====================================================

gate_probs = hetero_modt._gating_softmax(
    grid_modt,
    hetero_modt.theta_gating
)

expert_region = np.argmax(gate_probs, axis=1)

# =====================================================
# EXPERT PREDICTIONS
# =====================================================

tree = hetero_modt.DT_experts[0]
svm  = hetero_modt.DT_experts[1]
qda  = hetero_modt.DT_experts[2]

pred_tree = tree.predict(grid_modt)
pred_svm  = svm.predict(grid_modt)
pred_qda  = qda.predict(grid_modt)

# =====================================================
# COMBINE EXPERTS
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

# =====================================================
# RESHAPE
# =====================================================

final_pred = final_pred.reshape(xx.shape)
expert_region = expert_region.reshape(xx.shape)

# =====================================================
# NUMBER OF CLASSES
# =====================================================

n_classes = len(np.unique(y_train))

# =====================================================
# PLOT
# =====================================================

plt.figure(figsize=(9,8))

# Decision regions
plt.contourf(
    xx,
    yy,
    final_pred,
    levels=np.arange(n_classes + 1) - 0.5,
    alpha=0.30,
    cmap="viridis"
)

# Final decision boundaries
plt.contour(
    xx,
    yy,
    final_pred,
    levels=np.arange(n_classes - 1) + 0.5,
    colors="red",
    linewidths=2.5
)

# Gating boundaries
plt.contour(
    xx,
    yy,
    expert_region,
    levels=np.arange(hetero_modt.n_experts - 1) + 0.5,
    colors="black",
    linewidths=2
)

# Training samples
plt.scatter(
    X_train[:,0],
    X_train[:,1],
    c=y_train,
    cmap="viridis",
    s=10,
    edgecolors="k",
    linewidths=0.2,
    alpha=0.75
)

plt.title("Heterogeneous Mixture Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.gca().set_aspect("equal")
plt.tight_layout()




# =====================================================
# Expert predictions on grid
# =====================================================
x_min, x_max = X_train[:,0].min()-1, X_train[:,0].max()+1
y_min, y_max = X_train[:,1].min()-1, X_train[:,1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

grid = np.c_[xx.ravel(), yy.ravel()]

grid_modt = np.column_stack([
    grid,
    np.ones(len(grid))
])
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
# Mask outside each expert's region
# =====================================================

Z_tree_masked = np.ma.masked_where(
    ~tree_region,
    Z_tree
)

Z_svm_masked = np.ma.masked_where(
    ~svm_region,
    Z_svm
)

Z_qda_masked = np.ma.masked_where(
    ~qda_region,
    Z_qda
)

# =====================================================
# Number of classes
# =====================================================

n_classes = len(np.unique(y_train))

boundary_levels = np.arange(n_classes - 1) + 0.5

# =====================================================
# Plot
# =====================================================

plt.figure(figsize=(11, 9))

# -----------------------------------------------------
# Expert ownership regions
# -----------------------------------------------------

plt.contourf(
    xx,
    yy,
    expert_map,
    levels=np.arange(4) - 0.5,
    alpha=0.15,
    cmap="Set1"
)

# -----------------------------------------------------
# Gating boundaries
# -----------------------------------------------------

plt.contour(
    xx,
    yy,
    expert_map,
    levels=np.arange(3) + 0.5,
    colors="black",
    linewidths=2
)

# -----------------------------------------------------
# Tree boundaries
# -----------------------------------------------------

plt.contour(
    xx,
    yy,
    Z_tree_masked,
    levels=boundary_levels,
    linewidths=3,
    linestyles="solid",
    colors="blue"
)

# -----------------------------------------------------
# SVM boundaries
# -----------------------------------------------------

plt.contour(
    xx,
    yy,
    Z_svm_masked,
    levels=boundary_levels,
    linewidths=3,
    linestyles="dashed",
    colors="green"
)

# -----------------------------------------------------
# QDA boundaries
# -----------------------------------------------------

plt.contour(
    xx,
    yy,
    Z_qda_masked,
    levels=boundary_levels,
    linewidths=3,
    linestyles="dashdot",
    colors="red"
)

# -----------------------------------------------------
# Training data
# -----------------------------------------------------

plt.scatter(
    X_train[:,0],
    X_train[:,1],
    c=y_train,
    cmap="viridis",
    s=10,
    alpha=0.8
)

# -----------------------------------------------------
# Labels
# -----------------------------------------------------

from matplotlib.lines import Line2D

legend_lines = [
    Line2D([0],[0], color="blue",  lw=3, linestyle="solid"),
    Line2D([0],[0], color="green", lw=3, linestyle="dashed"),
    Line2D([0],[0], color="red",   lw=3, linestyle="dashdot"),
    Line2D([0],[0], color="black", lw=2)
]

plt.legend(
    legend_lines,
    [
        "Tree boundary",
        "SVM boundary",
        "QDA boundary",
        "Gating boundary"
    ]
)

plt.title(
    "Heterogeneous MoDT: Expert Boundaries Inside Their Assigned Regions"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()

plt.show()




# 2. MoDT Parameters (ORIGINAL)
# =========================================================
# 2. Train/Test Split
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

# 3. Train Original MoDT

# =========================================================

modt = MoDT(**parameters)
modt.fit(**parameters_fit)

# =========================================================
# 4. Evaluate
# =========================================================
print("Training accuracy:", modt.score_internal_disjoint())
test_acc = modt.score(X_test, y_test)
print("Test accuracy:", test_acc)


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
# MoDT predictions
# ==========================================

Z = modt.predict(grid)
Z = np.asarray(Z).reshape(xx.shape)

# ==========================================
# Number of classes
# ==========================================

n_classes = len(np.unique(y_train))

# ==========================================
# Plot
# ==========================================

plt.figure(figsize=(9, 8))

# decision regions
plt.contourf(
    xx,
    yy,
    Z,
    levels=np.arange(n_classes + 1) - 0.5,
    alpha=0.25,
    cmap="viridis"
)

# decision boundaries
plt.contour(
    xx,
    yy,
    Z,
    levels=np.arange(n_classes - 1) + 0.5,
    colors="black",
    linewidths=2
)

# training data
plt.scatter(
    X_train[:,0],
    X_train[:,1],
    c=y_train,
    cmap="viridis",
    s=10,
    alpha=0.7
)

plt.title("Homogeneous MoDT Decision Surface")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.gca().set_aspect("equal")
plt.tight_layout()


plt.show()







