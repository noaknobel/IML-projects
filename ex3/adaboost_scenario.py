import os
import pickle

import numpy as np
from typing import Tuple

from matplotlib import pyplot as plt

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adaboost import AdaBoost
from decision_stump import DecisionStump


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    # Train AdaBoost with 250 stumps
    n_learners = 250
    ab_model = AdaBoost(DecisionStump, iterations=n_learners)
    ab_model.fit(train_X, train_y)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ts = np.arange(1, n_learners + 1)
    train_err = [ab_model.partial_loss(train_X, train_y, t) for t in ts]
    test_err = [ab_model.partial_loss(test_X, test_y, t) for t in ts]
    plot_test_train_err(test_err, train_err, ts, noise)

    # Question 2: Plotting decision surfaces
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    comparing_ensemble_sizes(ab_model, test_X, test_y, lims, noise)

    # Question 3: Decision surface of best performing ensemble
    best_t = int(np.argmin(test_err)) + 1
    best_acc = 1 - test_err[best_t - 1]
    plot_best_ensemble_size(ab_model, best_acc, best_t, lims, test_X, test_y, noise)

    # Question 4: Decision surface with weighted samples
    plot_weighted_train(ab_model, lims, train_X, train_y, noise)


def plot_weighted_train(ab_model, lims, train_X, train_y, noise):
    D_final = ab_model.D_[-1]
    sizes = D_final / np.max(D_final) * 5
    fig4 = go.Figure([decision_surface(lambda X: ab_model.predict(X), lims[0], lims[1], showscale=False),
                      go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(size=sizes, color=train_y,
                                             symbol=[class_symbols[int((y_i + 1) // 2)] for y_i in train_y],
                                             colorscale=px.colors.diverging.RdBu, line=dict(color="black", width=1),
                                             sizemode="area"))])
    fig4.update_layout(title=f"Decision Surface with Training‐Point Weights (noise={noise})", xaxis=dict(range=lims[0], visible=False),
                       yaxis=dict(range=lims[1], visible=False), width=600, height=600)
    fig4.write_html(f"weighted_train_surface_noise{noise}.html", include_plotlyjs="cdn", full_html=True)


def plot_best_ensemble_size(ab_model, best_acc, best_t, lims, test_X, test_y, noise):
    fig = go.Figure([
        decision_surface(lambda X, T=best_t: ab_model.partial_predict(X, T), lims[0], lims[1], showscale=False),
        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                   marker=dict(color=test_y, symbol=[class_symbols[int((y_i + 1) // 2)] for y_i in test_y],
                               colorscale=px.colors.diverging.RdBu, line=dict(color="black", width=1), size=6))
    ], layout=go.Layout(width=500, height=500, xaxis=dict(visible=False), yaxis=dict(visible=False),
                        title=f"Best Ensemble Size: {best_t}, Accuracy: {best_acc:.2f} (noise={noise})"))
    fig.write_html(f"best_ensemble_surface_noise{noise}.html", include_plotlyjs="cdn", full_html=True)


def plot_test_train_err(test_err, train_err, ts, noise):
    plt.figure(figsize=(8, 5))
    plt.plot(ts, train_err, label="Train Error")
    plt.plot(ts, test_err, label="Test Error")
    plt.xlabel("Number of Weak Learners")
    plt.ylabel("Misclassification Error")
    plt.title(f"AdaBoost Train vs Test Error (noise={noise})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ada_train_test_error_noise{noise}.png", dpi=300, bbox_inches="tight")
    plt.close()


def comparing_ensemble_sizes(ab_model, test_X, test_y, lims, noise):
    T = [5, 50, 100, 250]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"T = {t}" for t in T], horizontal_spacing=0.05,
                        vertical_spacing=0.07)
    for i, t in enumerate(T):
        r, c = divmod(i, 2)
        row, col = r + 1, c + 1
        fig.add_traces([
            decision_surface(lambda X, T=t: ab_model.partial_predict(X, T), lims[0], lims[1], showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                       marker=dict(color=test_y, symbol=[class_symbols[int((y_i + 1) // 2)] for y_i in test_y],
                                   colorscale=px.colors.diverging.RdBu, line=dict(color="black", width=1), size=6))
        ], rows=row, cols=col)
        fig.update_xaxes(range=lims[0], row=row, col=col)
        fig.update_yaxes(range=lims[1], row=row, col=col)
    fig.update_layout(title_text=f"AdaBoost Partial Decision Surfaces (noise={noise})", height=800, width=800, )
    fig.write_html(f"adaboost_surfaces_noise{noise}.html", include_plotlyjs="cdn", full_html=True)


if __name__ == '__main__':
    np.random.seed(0)
    for noise in [0.0, 0.4]:
        fit_and_evaluate_adaboost(noise)
