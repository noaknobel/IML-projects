import pandas as pd
from matplotlib import pyplot as plt


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
    df = df[df["Temp"] > -72.77777]

    # Add DayOfYear column based on parsed dates
    df["DayOfYear"] = df["Date"].dt.dayofyear
    return df


def explore_country(df: pd.DataFrame):
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
    plt.show()

    # Plot 2
    monthly_std = israel_df.groupby("Month")["Temp"].std()
    monthly_std.plot(kind="bar", color="skyblue")
    plt.title("Standard Deviation of Daily Temperatures by Month (Israel)")
    plt.xlabel("Month")
    plt.ylabel("Temperature Std Dev (°C)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country
    explore_country(df)

    # Question 4 - Exploring differences between countries

    # Question 5 - Fitting model for different values of `k`

    # Question 6 - Evaluating fitted model on different countries
    pass