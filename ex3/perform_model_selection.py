import numpy as np
import plotly.graph_objects as go
from sklearn import datasets
from cross_validate import cross_validate
from estimators import RidgeRegression, Lasso, LinearRegression


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    perm = np.random.permutation(X.shape[0])
    train_idx, test_idx = perm[:n_samples], perm[n_samples:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Question 2 - Perform Cross Validation for different values of the regularization parameter for Ridge and
    # Lasso regressions
    ridge_grid = np.linspace(1e-5, 5e-2, n_evaluations)
    lasso_grid = np.linspace(1e-2, 2.0, n_evaluations)

    ridge_train, ridge_val = [], []
    lasso_train, lasso_val = [], []

    for lam in ridge_grid:
        ridge = RidgeRegression(lam=lam, include_intercept=True)
        tr, va = cross_validate(ridge, X_train, y_train, cv=5)
        ridge_train.append(tr)
        ridge_val.append(va)

    for alpha in lasso_grid:
        lasso = Lasso(alpha=alpha, include_intercept=True)
        tr, va = cross_validate(lasso, X_train, y_train, cv=5)
        lasso_train.append(tr)
        lasso_val.append(va)

    plot_cross_valdition(lasso_grid, lasso_train, lasso_val, ridge_grid, ridge_train, ridge_val)

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    # Find the best λ’s
    best_lam = ridge_grid[np.argmin(ridge_val)]
    best_alpha = lasso_grid[np.argmin(lasso_val)]
    print(f"Best Ridge λ = {best_lam:.4g}")
    print(f"Best Lasso λ = {best_alpha:.4g}")

    # Retrain on the full 50‐sample train set
    best_ridge = RidgeRegression(lam=best_lam, include_intercept=True)
    best_ridge.fit(X_train, y_train)

    best_lasso = Lasso(alpha=best_alpha, include_intercept=True)
    best_lasso.fit(X_train, y_train)

    plain_ls = LinearRegression()
    plain_ls.fit(X_train, y_train)

    print(f"Test MSE Ridge = {best_ridge.loss(X_test, y_test):.3f}")
    print(f"Test MSE Lasso = {best_lasso.loss(X_test, y_test):.3f}")
    print(f"Test MSE Least-Squares = {plain_ls.loss(X_test, y_test):.3f}")


def plot_cross_valdition(lasso_grid, lasso_train, lasso_val, ridge_grid, ridge_train, ridge_val):
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=ridge_grid, y=ridge_train, mode='lines', name='Train'))
    fig_r.add_trace(go.Scatter(x=ridge_grid, y=ridge_val, mode='lines', name='Validation'))
    fig_r.update_xaxes(title_text='λ')
    fig_r.update_yaxes(title_text='MSE')
    fig_r.update_layout(
        title_text='Ridge Cross‐Validation (λ ∈ [1e-5, 5e-2])',
        width=700, height=400
    )
    fig_r.write_html(
        'cv_ridge_linear.html',
        include_plotlyjs='cdn', full_html=True
    )
    fig_l = go.Figure()
    fig_l.add_trace(go.Scatter(x=lasso_grid, y=lasso_train, mode='lines', name='Train'))
    fig_l.add_trace(go.Scatter(x=lasso_grid, y=lasso_val, mode='lines', name='Validation'))
    fig_l.update_xaxes(title_text='λ')
    fig_l.update_yaxes(title_text='MSE')
    fig_l.update_layout(
        title_text='Lasso Cross‐Validation (λ ∈ [1e-2, 2])',
        width=700, height=400
    )
    fig_l.write_html(
        'cv_lasso_linear.html',
        include_plotlyjs='cdn', full_html=True
    )


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()