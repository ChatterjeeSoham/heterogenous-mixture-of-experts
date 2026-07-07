import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Dataset Generator
# =========================================================
def generate_blob_ring_dataset(n=500, inner_radius=2.5, seed=42):

    np.random.seed(seed)

    # Number of samples per class
    n_per_class = n // 2

    # Class 0: Inner blob
    mean0 = [0, 0]
    cov0 = [[1.5, 0],
            [0, 1.5]]

    X0 = np.random.multivariate_normal(mean0, cov0, n_per_class)

    # Class 1: Outer spread
    mean1 = [0, 0]
    cov1 = [[15, 0],
            [0, 15]]

    X1 = np.random.multivariate_normal(mean1, cov1, n_per_class)

    # Keep only outer points to form a ring
    distances = np.linalg.norm(X1, axis=1)
    X1 = X1[distances > inner_radius]

    # Labels
    y0 = np.zeros(len(X0), dtype=int)
    y1 = np.ones(len(X1), dtype=int)

    # Final dataset
    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1))

    return X, y
# =====================================================
# Example usage
# =====================================================

if __name__ == "__main__":

    X, y = generate_blob_ring_dataset()

    plt.figure(figsize=(6,6))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="tab10",
        s=12
    )

    plt.title("Inner Blob + Outer Ring")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.show()