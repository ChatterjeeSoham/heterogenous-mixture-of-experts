import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Dataset Generator
# =========================================================
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

# =====================================================
# Example usage
# =====================================================

if __name__ == "__main__":

    X, y = generate_piecewise_linear_kink()

    plt.figure(figsize=(6,6))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="tab10",
        s=12
    )

    plt.title("Piecewise Linear Kink")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.show()
