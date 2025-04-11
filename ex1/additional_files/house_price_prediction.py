from typing import NoReturn

import pandas as pd
from matplotlib import pyplot as plt


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
    # Make a copy so we don't modify the original
    X = X.copy()
    y = y.copy()
    print(len(X))

    # Remove rows with invalid values (like 0 bedrooms or bathrooms)
    invalid_rows = (X["bedrooms"] == 0) | (X["bathrooms"] == 0)
    X = X[~invalid_rows]
    y = y[~invalid_rows]
    print(len(X))

    # Remove rows where year of renovation is before construction
    invalid_years = (X["yr_renovated"] != 0) & (X["yr_renovated"] < X["yr_built"])
    print(f"Removing {invalid_years.sum()} rows with renovation year before build year")
    X = X[~invalid_years]
    y = y[~invalid_years]

    X["date"] = pd.to_datetime(X["date"], format="%Y%m%dT%H%M%S")
    X["sale_year"] = X["date"].dt.year
    invalid_sale_year = (X["sale_year"] < X["yr_built"]) | \
                        ((X["yr_renovated"] != 0) & (X["sale_year"] < X["yr_renovated"]))
    print(f"Removing {invalid_sale_year.sum()} rows where sale date is before build or renovation year")
    X = X[~invalid_sale_year]
    y = y[~invalid_sale_year]

    invalid_dates = (X["sale_year"] > 2015) | (X["yr_renovated"] > 2015) | (X["yr_built"] > 2015)
    print(f"Found {invalid_dates.sum()} rows with future dates")
    X = X[~invalid_dates]
    y = y[~invalid_dates]

    invalid_sqft = X["sqft_living"] > X["sqft_lot"]
    print(f"Found {invalid_sqft.sum()} rows where sqft_living > sqft_lot")
    X = X[~invalid_sqft]
    y = y[~invalid_sqft]

    lot_diff = abs(X["sqft_lot"] - X["sqft_lot15"])
    living_diff = abs(X["sqft_living"] - X["sqft_living15"])
    lot_threshold = 0.9
    living_threshold = 0.9

    invalid_lot = lot_diff > lot_threshold * X["sqft_lot"]
    invalid_living = living_diff > living_threshold * X["sqft_living"]

    # Combine the conditions and filter the data
    invalid_diff = invalid_lot | invalid_living
    print(f"Found {invalid_diff.sum()} rows with large differences")
    X = X[~invalid_diff]
    y = y[~invalid_diff]

    # Create new features that might help:
    X["house_age"] = 2025 - X["yr_built"]
    X["renovated"] = X["yr_renovated"].apply(lambda x: 0 if x == 0 else 1)
    X["years_since_renovation"] = X.apply(
        lambda row: 0 if row["yr_renovated"] == 0 else 2025 - row["yr_renovated"], axis=1)

    # drop yr_built and yr_renovated after using them
    X.drop(columns=["yr_built", "yr_renovated"], inplace=True)

    # Drop irrelevant features
    X.drop(columns=["id", "date"], inplace=True)

    # Reset index after dropping
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    print(len(X))
    return X, y


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
    # Create new features that might help:
    X["house_age"] = 2025 - X["yr_built"]
    X["renovated"] = X["yr_renovated"].apply(lambda x: 0 if x == 0 else 1)
    X["years_since_renovation"] = X.apply(
        lambda row: 0 if row["yr_renovated"] == 0 else 2025 - row["yr_renovated"], axis=1)

    # drop yr_built and yr_renovated after using them
    X.drop(columns=["yr_built", "yr_renovated"], inplace=True)

    # Drop irrelevant features
    X.drop(columns=["id", "date"], inplace=True)

    # Reset index after dropping
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
        # Calculate covariance between feature and response
        covariance = X[feature].cov(y)

        # Calculate standard deviations of the feature and the response
        std_feature = X[feature].std()
        std_response = y.std()

        # Calculate Pearson correlation coefficient
        pearson_corr = covariance / (std_feature * std_response)

        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(X[feature], y, alpha=0.6, color='blue')
        plt.title(f"{feature} vs. Price\nPearson Correlation: {pearson_corr:.3f}")
        plt.xlabel(feature)
        plt.ylabel("Price")

        # Save the plot to the output path with the feature name in the filename
        plt.savefig(f"{output_path}/{feature}_vs_price.png")
        plt.close()


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    # Split into train/test sets (75/25)
    split_index = int(0.75 * len(df))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    # Separate features and target
    X_train, y_train = train_df.drop("price", axis=1), train_df.price
    X_test, y_test = test_df.drop("price", axis=1), test_df.price

    # Question 3 - preprocessing of housing prices train dataset
    X_train, y_train = preprocess_train(X_train, y_train)

    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_train, y_train, output_path="feature_plots")

    # Question 5 - preprocess the test data
    X_test = preprocess_test(X_test)

    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
