import os

import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt

from polynomial_fitting import PolynomialFitting


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # Load the dataset with date parsing
    df = pd.read_csv(filename, parse_dates=["Date"])

    # Drop invalid rows
    df.dropna(inplace=True)
    df = df[round(df["Temp"], 2) != -72.78]

    # Add DayOfYear column based on parsed dates
    df["DayOfYear"] = df["Date"].dt.dayofyear
    return df


def explore_israel(df: pd.DataFrame, output_path: str = "."):
    # Filter for Israel only
    israel_df = df[df["Country"] == "Israel"].copy()
    # Plot 1
    for year in israel_df["Year"].unique():
        year_df = israel_df[israel_df["Year"] == year]
        plt.scatter(year_df["DayOfYear"], year_df["Temp"], label=str(year), s=10)
    plt.title("Average Daily Temperature in Israel by Day of Year (Colored by Year)")
    plt.xlabel("Day of Year")
    plt.ylabel("Temperature (°C)")
    plt.legend(title="Year", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    output_file = os.path.join(output_path, "temperature_by_day.png")
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(output_file)
    plt.close()

    # Plot 2
    monthly_std = israel_df.groupby("Month")["Temp"].std()
    monthly_std.plot(kind="bar", color="skyblue")
    plt.title("Standard Deviation of Daily Temperatures by Month (Israel)")
    plt.xlabel("Month")
    plt.ylabel("Temperature Std Dev (°C)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    output_file = os.path.join(output_path, "temperature_by_month.png")
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(output_file)
    plt.close()


def explore_per_country(df: pd.DataFrame, output_path: str = "."):
    grouped = df.groupby(["Country", "Month"])["Temp"].agg(["mean", "std"]).reset_index()
    grouped.rename(columns={"mean": "AverageTemp", "std": "TempStd"}, inplace=True)

    # Plot using Plotly Express with error bars
    fig = px.line(
        grouped,
        x="Month",
        y="AverageTemp",
        color="Country",
        error_y="TempStd",
        markers=True,
        title="Average Monthly Temperature with Standard Deviation per Country"
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Temperature (°C)")
    output_file = os.path.join(output_path, "monthly_temperature_per_country.html")
    os.makedirs(output_path, exist_ok=True)
    fig.write_html(output_file)


def evaluate_polynomial_degrees(df: pd.DataFrame, random_state: int = 0, output_path: str = "."):
    """
    Evaluate polynomial fitting for Israel data using polynomial degrees 1 to 10.
    """
    # Filter only Israel data and shuffle
    israel_df = df[df["Country"] == "Israel"].copy()
    israel_df = israel_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # 75/25 split
    split_idx = int(0.75 * len(israel_df))
    train_df = israel_df.iloc[:split_idx]
    test_df = israel_df.iloc[split_idx:]

    # Extract features and targets
    X_train = train_df["DayOfYear"].to_numpy()
    y_train = train_df["Temp"].to_numpy()
    X_test = test_df["DayOfYear"].to_numpy()
    y_test = test_df["Temp"].to_numpy()

    test_errors = []

    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(X_train, y_train)
        loss = model.loss(X_test, y_test)
        rounded_loss = round(loss, 2)
        test_errors.append(rounded_loss)
        print(f"Test error for polynomial degree {k}: {rounded_loss}")

    # Bar plot of test errors
    plt.bar(range(1, 11), test_errors, color="skyblue")
    plt.xlabel("Polynomial Degree (k)")
    plt.ylabel("Test MSE Loss")
    plt.title("Test Error vs. Polynomial Degree")
    plt.xticks(range(1, 11))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    output_file = os.path.join(output_path, "test_error_vs_polynomial_degree.png")
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(output_file)
    plt.close()

    # Report best k
    min_loss = min(test_errors)
    return next(k for k, loss in enumerate(test_errors, 1) if loss == min_loss)


def evaluate_model_on_other_countries(df: pd.DataFrame, best_k: int, output_path: str = "."):
    """
    Fit a polynomial model on all Israel data using the chosen degree,
    and evaluate its test loss on each of the other countries.
    """
    # Select Israel data for training
    israel_df = df[df["Country"] == "Israel"]
    X_israel = israel_df["DayOfYear"].to_numpy()
    y_israel = israel_df["Temp"].to_numpy()

    # Normalize using Israel's day-of-year range
    x_min, x_max = X_israel.min(), X_israel.max()
    X_israel_norm = (X_israel - x_min) / (x_max - x_min)

    # Fit model
    model = PolynomialFitting(best_k)
    model.fit(X_israel_norm, y_israel)

    # Evaluate on other countries
    other_countries = df["Country"].unique()
    other_countries = [c for c in other_countries if c != "Israel"]

    errors = []
    for country in other_countries:
        country_df = df[df["Country"] == country]
        X_country = country_df["DayOfYear"].to_numpy()
        y_country = country_df["Temp"].to_numpy()

        # Normalize using Israel's scale!
        X_country_norm = (X_country - x_min) / (x_max - x_min)

        loss = model.loss(X_country_norm, y_country)
        errors.append((country, round(loss, 2)))

    # Plot
    countries, losses = zip(*errors)
    plt.figure(figsize=(7, 5))
    plt.bar(countries, losses, color="skyblue")
    plt.title(f"Test Error on Other Countries (Model Trained on Israel, k={k})")
    plt.ylabel("MSE Loss")
    plt.xlabel("Country")
    plt.tight_layout()
    output_file = os.path.join(output_path, "test_error_on_other_country.png")
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(output_file)
    plt.close()


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country
    explore_israel(df, output_path="city_temperature_plots")

    # Question 4 - Exploring differences between countries
    explore_per_country(df, output_path="city_temperature_plots")

    # Question 5 - Fitting model for different values of `k`
    k = evaluate_polynomial_degrees(df, output_path="city_temperature_plots")

    # Question 6 - Evaluating fitted model on different countries
    evaluate_model_on_other_countries(df, k, output_path="city_temperature_plots")
