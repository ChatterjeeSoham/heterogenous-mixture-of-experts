import numpy as np
import matplotlib.pyplot as plt

def generate_hierarchical_gaussians(
    n_per_class=500,
    noise=0.35,
    seed=42
):

    np.random.seed(seed)

    X_list = []
    y_list = []

    # =====================================================
    # SUPERCLUSTER 1
    # Classes 0,1,2
    # =====================================================

    centers_1 = [

        (-10,  8),
        (-6,   4),
        (-12,  2)
    ]

    # =====================================================
    # SUPERCLUSTER 2
    # Classes 3,4,5
    # =====================================================

    centers_2 = [

        (8,  8),
        (12, 4),
        (6,  2)
    ]

    all_centers = centers_1 + centers_2

    # =====================================================
    # Generate each subclass
    # =====================================================

    for k, (cx, cy) in enumerate(all_centers):

        # Different covariance structures
        angle = (k + 1) * np.pi / 10

        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])

        D = np.array([
            [2.5, 0.0],
            [0.0, 0.6]
        ])

        cov = R @ D @ R.T

        Xk = np.random.multivariate_normal(
            mean=[cx, cy],
            cov=cov,
            size=n_per_class
        )

        # Add extra local noise
        Xk += noise * np.random.randn(*Xk.shape)

        yk = np.full(n_per_class, k)

        X_list.append(Xk)
        y_list.append(yk)

    # =====================================================
    # Add bridge overlap between subclasses
    # =====================================================

    for k in range(len(X_list)):

        if k < 3:

            X_list[k][:, 0] += (
                0.6 * np.sin(X_list[k][:, 1] / 2)
            )

        else:

            X_list[k][:, 1] += (
                0.6 * np.cos(X_list[k][:, 0] / 2)
            )

    # =====================================================
    # Combine dataset
    # =====================================================

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    return X, y
# =====================================================
# Example usage
# =====================================================

if __name__ == "__main__":

    X, y = generate_hierarchical_gaussians()

    plt.figure(figsize=(6,6))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="tab10",
        s=12
    )

    plt.title("Hierarchical gaussians")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.show()
