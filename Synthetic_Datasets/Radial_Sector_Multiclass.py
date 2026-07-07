import numpy as np
import matplotlib.pyplot as plt


def generate_radial_sector_dataset(
    n_per_class=500,
    noise=0.25,
    seed=42
):

    np.random.seed(seed)

    X_list = []
    y_list = []

    # =====================================================
    # Define sectors
    # =====================================================

    sector_centers = [

        0,
        np.pi / 3,
        2 * np.pi / 3,
        np.pi,
        4 * np.pi / 3,
        5 * np.pi / 3
    ]

    # =====================================================
    # Generate each class
    # =====================================================

    for k, center_angle in enumerate(sector_centers):

        # Angular spread
        theta = np.random.normal(
            loc=center_angle,
            scale=0.30,
            size=n_per_class
        )

        # Radial spread
        r = np.random.uniform(
            low=3,
            high=9,
            size=n_per_class
        )

        # Add nonlinear radial perturbation
        r += (
            0.8 * np.sin(3 * theta)
        )

        # Convert polar to Cartesian
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Add Gaussian noise
        x += noise * np.random.randn(n_per_class)
        y += noise * np.random.randn(n_per_class)

        Xk = np.column_stack((x, y))

        yk = np.full(n_per_class, k)

        X_list.append(Xk)
        y_list.append(yk)

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

    X, y = generate_radial_sector_dataset()

    plt.figure(figsize=(6,6))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="tab10",
        s=12
    )

    plt.title("Radial Sector Dataset")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.show()
