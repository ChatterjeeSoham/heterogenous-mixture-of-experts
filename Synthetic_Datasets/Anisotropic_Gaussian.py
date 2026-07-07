import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Dataset Generator
# =========================================================
def generate_Anisotropic_Gaussian(n=300, seed=42):

    np.random.seed(seed)

    # Class 0: Horizontal spread
    X1 = np.random.multivariate_normal(
        mean=[0, 0],
        cov=[[5.0, 0.0],
             [0.0, 0.1]],
        size=n
    )

    # Class 1: Vertical spread
    X2 = np.random.multivariate_normal(
        mean=[0, 0],
        cov=[[0.1, 0.0],
             [0.0, 5.0]],
        size=n
    )

    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(n, dtype=int),
                   np.ones(n, dtype=int)))

    return X, y
# =====================================================
# Example usage
# =====================================================

if __name__ == "__main__":

    X, y = generate_Anisotropic_Gaussian()

    plt.figure(figsize=(6,6))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="tab10",
        s=12
    )

    plt.title("Anisotropic Gaussian")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.show()