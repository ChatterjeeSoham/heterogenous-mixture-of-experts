import numpy as np
import matplotlib.pyplot as plt


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
# =====================================================
# Example usage
# =====================================================

if __name__ == "__main__":

    X, y = generate_multiclass_anisotropic_gaussians()

    plt.figure(figsize=(6,6))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="tab10",
        s=12
    )

    plt.title("Multiclass Anisotropic Gaussian")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.show()
