import os
from typing import NoReturn

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from linear_regression import LinearRegression


def split_train_test(df: pd.DataFrame, test_ratio: float = 0.25, random_state: int = 0):
    """
    Shuffle and split the dataset into train and test sets.
    """
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_index = int((1 - test_ratio) * len(df))
    train_df, test_df = df.iloc[:split_index], df.iloc[split_index:]

    X_train, y_train = train_df.drop("price", axis=1), train_df.price
    X_test, y_test = test_df.drop("price", axis=1), test_df.price

    return X_train, y_train, X_test, y_test


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    X, y = X.copy(), y.copy()

    X, y = _remove_invalid_rows(X, y)
    X, y = _remove_invalid_dates(X, y)
    X, y = _filter_large_differences(X, y)
    X = _create_features(X)

    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    return X, y


def _remove_invalid_rows(X, y):
    invalid = (X["bedrooms"] == 0) | (X["bathrooms"] == 0) | (np.isnan(y))
    X, y = X[~invalid], y[~invalid]
    return X, y


def _remove_invalid_dates(X, y):
    # Remove inconsistent renovation years
    invalid_renovation = (X["yr_renovated"] != 0) & (X["yr_renovated"] < X["yr_built"])
    X, y = X[~invalid_renovation], y[~invalid_renovation]

    # Parse date and extract sale year
    X["date"] = pd.to_datetime(X["date"], format="%Y%m%dT%H%M%S")
    X["sale_year"] = X["date"].dt.year

    # Remove inconsistent sale years
    invalid_sale = (X["sale_year"] < X["yr_built"]) | \
                   ((X["yr_renovated"] != 0) & (X["sale_year"] < X["yr_renovated"]))
    X, y = X[~invalid_sale], y[~invalid_sale]

    # Remove future dates
    future = (X["sale_year"] > 2015) | (X["yr_renovated"] > 2015) | (X["yr_built"] > 2015)
    X.drop(columns=["sale_year"], inplace=True)
    return X[~future], y[~future]


def _filter_large_differences(X, y):
    lot_diff = abs(X["sqft_lot"] - X["sqft_lot15"])
    living_diff = abs(X["sqft_living"] - X["sqft_living15"])
    threshold = 2.5

    invalid = (lot_diff / threshold > X["sqft_lot"]) | (living_diff / threshold > X["sqft_living"])
    return X[~invalid], y[~invalid]


def _create_features(X):
    X["house_age"] = 2025 - X["yr_built"]
    X["renovated"] = X["yr_renovated"].apply(lambda x: 0 if x == 0 else 1)

    # Drop used/irrelevant columns
    X.drop(columns=["yr_built", "yr_renovated", "id", "date"], inplace=True)
    return X


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    X = _create_features(X)
    X.reset_index(drop=True, inplace=True)
    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X.columns:
        covariance = X[feature].cov(y)
        std_feature = X[feature].std()
        std_response = y.std()
        pearson_corr = covariance / (std_feature * std_response)

        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(X[feature], y, alpha=0.6, color='blue')
        plt.title(f"{feature} vs. Price\nPearson Correlation: {pearson_corr:.3f}")
        plt.xlabel(feature)
        plt.ylabel("Price")
        output_file = os.path.join(output_path, f"{feature}_vs_price.png")
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(output_file)
        plt.close()


def evaluate_training_size_effect(X_train: pd.DataFrame, y_train: pd.Series,
                                  X_test: pd.DataFrame, y_test: pd.Series,
                                  output_path: str = "."):
    percentages = range(10, 101)
    mean_losses = []
    std_losses = []

    X_test_np, y_test_np = X_test.to_numpy(), y_test.to_numpy()

    for p in percentages:
        losses = []
        for _ in range(10):
            sample = X_train.sample(frac=p / 100.0)
            sample_y = y_train.loc[sample.index]
            sample_X_np = sample.to_numpy()
            sample_y_np = sample_y.to_numpy()
            # Train and evaluate
            model = LinearRegression(include_intercept=True)
            model.fit(sample_X_np, sample_y_np)
            loss = model.loss(X_test_np, y_test_np)
            losses.append(loss)
        mean_losses.append(np.mean(losses))
        std_losses.append(np.std(losses))

    percentages = np.array(percentages)
    mean_losses = np.array(mean_losses)
    std_losses = np.array(std_losses)

    plt.figure(figsize=(10, 6))
    plt.plot(percentages, mean_losses, label="Mean Test Loss", color='blue')
    plt.fill_between(percentages,
                     mean_losses - 2 * std_losses,
                     mean_losses + 2 * std_losses,
                     color='blue', alpha=0.2, label="±2 Standard Deviations")
    plt.xlabel("Percentage of Training Data Used")
    plt.ylabel("Mean Squared Error on Test Set")
    plt.title("Test MSE vs. Training Set Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_file = os.path.join(output_path, "training_size_vs_loss.png")
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(output_file)


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")

    # Question 2 - split train test
    X_train, y_train, X_test, y_test = split_train_test(df)

    # Question 3 - preprocessing of housing prices train dataset
    X_train, y_train = preprocess_train(X_train, y_train)

    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_train, y_train, output_path="house_price_plots")

    # Question 5 - preprocess the test data
    X_test = preprocess_test(X_test)

    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    evaluate_training_size_effect(X_train, y_train, X_test, y_test, output_path="house_price_plots")
