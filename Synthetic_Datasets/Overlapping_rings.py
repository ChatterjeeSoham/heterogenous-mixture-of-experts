import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Dataset Generator
# =========================================================
def generate_overlapping_rings(n=3000, noise=0.35, r1=1.4, r2=2.0, seed=42):
    """
    Generate two noisy concentric rings with overlap.
    """

    np.random.seed(seed)
    n2 = n // 2

    theta1 = 2 * np.pi * np.random.rand(n2)
    theta2 = 2 * np.pi * np.random.rand(n2)

    R1 = r1 + noise * np.random.randn(n2)
    R2 = r2 + noise * np.random.randn(n2)

    X1 = np.c_[R1 * np.cos(theta1), R1 * np.sin(theta1)]
    X2 = np.c_[R2 * np.cos(theta2), R2 * np.sin(theta2)]

    X = np.vstack([X1, X2])
    y = np.array([0] * n2 + [1] * n2)

    return X, y
# =====================================================
# Example usage
# =====================================================

if __name__ == "__main__":

    X, y = generate_overlapping_rings()

    plt.figure(figsize=(6,6))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="tab10",
        s=12
    )

    plt.title("Overlapping Rings")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.show()
