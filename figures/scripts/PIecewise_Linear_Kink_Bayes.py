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
plt.savefig("myplot.png", dpi=300, bbox_inches="tight")
plt.show()