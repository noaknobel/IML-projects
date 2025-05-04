import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


### HELPER FUNCTIONS ###

# Function to generate labels based on f(x)
def generate_labels(X):
    weights = np.array([-0.6, 0.4])
    return np.sign(np.dot(X, weights))


# Function to generate dataset
def generate_data(m):
    while True:
        X = np.random.multivariate_normal([0, 0], np.array([[1, 0.5], [0.5, 1]]), m)
        y = generate_labels(X)
        # Check if both classes are present
        if len(np.unique(y)) > 1:  # At least 2 classes are present
            break  # If both classes are present, break out of the loop
    return X, y


def run_svm(X, y, C_value):
    model = SVC(C=C_value, kernel='linear')
    model.fit(X, y)  # Train on the entire dataset
    return model


def plot_decision_boundary_lines(X, y, model, ax, title):
    # Plot f decision boundary
    weights = np.array([-0.6, 0.4])
    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_vals_true = -(weights[0] * x_vals) / weights[1]  # Equation of line: x1 = -w1/w2 * x2
    ax.plot(x_vals, y_vals_true, 'k--', label="f Boundary")  # True decision boundary line

    # Plot SVM decision boundary
    slope = -model.coef_[0][0] / model.coef_[0][1]
    intercept = -model.intercept_[0] / model.coef_[0][1]
    y_vals_svm = slope * x_vals + intercept
    ax.plot(x_vals, y_vals_svm, 'r-', label="SVM Boundary")  # SVM decision boundary line

    # Scatter plot of data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, marker='o', cmap=plt.cm.RdBu)
    ax.set_title(title)
    ax.legend()


def generate_data_2():
    # Moons dataset
    X_moons, y_moons = make_moons(n_samples=200, noise=0.2)

    # Circles dataset
    X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.5)

    # Two Gaussians dataset
    mean1 = [-1, -1]
    mean2 = [1, 1]
    cov = np.array([[0.5, 0.2], [0.2, 0.5]])
    X_gaussians = np.vstack([np.random.multivariate_normal(mean1, cov, 100),
                             np.random.multivariate_normal(mean2, cov, 100)])
    y_gaussians = np.array([0] * 100 + [1] * 100)

    X_moons_train, X_moons_test, y_moons_train, y_moons_test = train_test_split(X_moons, y_moons, test_size=0.2,
                                                                                random_state=42)
    X_circles_train, X_circles_test, y_circles_train, y_circles_test = train_test_split(X_circles, y_circles,
                                                                                        test_size=0.2, random_state=42)
    X_gaussians_train, X_gaussians_test, y_gaussians_train, y_gaussians_test = train_test_split(X_gaussians,
                                                                                                y_gaussians,
                                                                                                test_size=0.2,
                                                                                                random_state=42)
    return (X_moons_train, X_moons_test, y_moons_train, y_moons_test), (
        X_circles_train, X_circles_test, y_circles_train, y_circles_test), (
        X_gaussians_train, X_gaussians_test, y_gaussians_train, y_gaussians_test)


def get_classifiers():
    svm = SVC(C=1 / 5, kernel='linear')  # SVM with λ = 5
    dtree = DecisionTreeClassifier(max_depth=7)  # Decision Tree with depth 7
    knn = KNeighborsClassifier(n_neighbors=5)  # KNN with k = 5
    return svm, dtree, knn


def plot_decision_boundary(X, y, model, title, path=None):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.spring)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', cmap=plt.cm.spring, edgecolors='k')
    plt.title(title)
    save_plot(path)


def save_plot(file_path):
    if file_path:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory) and directory != '':
            os.makedirs(directory)
        plt.savefig(file_path)
    else:
        plt.show()
    plt.close()


### Exercise Solution ###

def pratical_1_runner(save_path=None):
    m_values = [5, 10, 20, 100]
    C_values = [0.1, 1, 5, 10, 100]

    for m in m_values:
        X, y = generate_data(m)
        for C in C_values:
            model = run_svm(X, y, C)  # Train on all points
            title = f"m={m}, C={C}"
            file_name = f"svm_m{m}_C{C}.png"
            file_path = os.path.join(save_path, file_name) if save_path else None
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_decision_boundary_lines(X, y, model, ax, title)
            save_plot(file_path)


def practical_2_runner(save_path=None):
    (X_moons_train, X_moons_test, y_moons_train, y_moons_test), \
        (X_circles_train, X_circles_test, y_circles_train, y_circles_test), \
        (X_gaussians_train, X_gaussians_test, y_gaussians_train, y_gaussians_test) = generate_data_2()
    classifiers = get_classifiers()
    datasets = [("Moons", X_moons_train, y_moons_train, X_moons_test, y_moons_test),
                ("Circles", X_circles_train, y_circles_train, X_circles_test, y_circles_test),
                ("Two Gaussians", X_gaussians_train, y_gaussians_train, X_gaussians_test, y_gaussians_test)]

    for dataset_name, X_train, y_train, X_test, y_test in datasets:
        for clf, name in zip(classifiers, ['SVM', 'Decision Tree', 'KNN']):
            clf.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, clf.predict(X_test))
            title = f'{name} (Accuracy: {accuracy:.2f})'
            file_name = f'{dataset_name}_{name}_boundary.png'
            file_path = os.path.join(save_path, file_name) if save_path else None
            plot_decision_boundary(np.vstack([X_train, X_test]), np.hstack([y_train, y_test]), clf, title, file_path)


if __name__ == "__main__":
    path = None
    # pratical_1_runner(save_path="first_part_plots")
    practical_2_runner(save_path="second_part_plots")
