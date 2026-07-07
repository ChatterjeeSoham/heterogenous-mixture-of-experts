import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Dataset Generator
# =========================================================
def generate_Rotated_Gaussian_Slabs(n=3000, noise=0.6, angle=np.pi/4, seed=42):

    np.random.seed(seed)
    n2 = n // 2

    X0 = np.random.randn(n2, 2)
    X1 = np.random.randn(n2, 2)

    # Stretch and shift
    X0[:, 0] *= 3.0
    X1[:, 0] *= 3.0
    X0[:, 0] -= 2.0
    X1[:, 0] += 2.0

    X = np.vstack([X0, X1])
    y = np.array([0]*n2 + [1]*n2)

    # Rotate
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    X = X @ R.T

    # Add noise
    X += noise * np.random.randn(*X.shape)

    return X, y
# =====================================================
# Example usage
# =====================================================

if __name__ == "__main__":

    X, y = generate_Rotated_Gaussian_Slabs()

    plt.figure(figsize=(6,6))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="tab10",
        s=12
    )

    plt.title("Rotated Gaussian Slabs")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.show()
