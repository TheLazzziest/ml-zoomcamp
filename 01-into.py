import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    return mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CRISP-DM

    It's a lifecycle management framework applied for data centric applications

    ## Lifecycle
    """)
    return


@app.cell
def _(mo):
    mo.mermaid(r"""
    flowchart LR
        A(Business<br>Understanding) --> B(Data<br>Understanding)
        B --> C(Data<br>Preparation)
        C --> D(Modeling)
        D --> E(Evaluation)
        E --> F(Deployment)
        F --> A

        classDef phase fill:#f9f9f9,stroke:#333,stroke-width:2px;
        class A,B,C,D,E,F phase;
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Business Understanding

    Define business objectives, success criteria, resources, risks, and a project plan.

    ## Data Understanding

    Collect, describe, explore, and assess data quality (completeness, correctness, patterns).

    ## Data Preparation

    Select, clean, integrate, transform, and format data for modeling (often ~80% of project time).

    ## Modeling

    Select algorithms (regression, classification, clustering, etc.), build models, and tune parameters.

    ## Evaluation

    Assess whether models meet business success criteria; review the process and decide next steps.

    ## Deployment

    Integrate the approved model into production (reports, APIs, automated scoring) and plan ongoing monitoring.
    """)
    return


@app.cell
def _(pd):
    # Check pandas version
    pd.__version__
    return


@app.cell
def _(pd):
    # Ingest the data
    df = pd.read_csv("https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv")

    df.shape
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    # Count missing values
    df.isnull().sum()
    return


@app.cell
def _(df):
    # Median value of horsepower
    # Find the median value of the horsepower column in the dataset.
    # Next, calculate the most frequent value of the same horsepower column.
    # Use the fillna method to fill the missing values in the horsepower column with the most frequent value from the previous step.
    # Now, calculate the median value of horsepower once again.
    (
        df.horsepower.agg("median") - df.horsepower.fillna(
            df.horsepower.agg("mode")
        ).agg("median")
    )


    return


@app.cell
def _(df, np):
    # sum of weights
    # Select all the cars from Asia
    # Select only columns vehicle_weight and model_year
    # Select the first 7 values
    # Get the underlying NumPy array. Let's call it X.
    # Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
    # Invert XTX.
    # Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200].
    # Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
    # What's the sum of all the elements of the result?
    asian_df = df[df["origin"] == "Asia"][["vehicle_weight", "model_year"]].head(7)
    XTX = np.dot(asian_df.to_numpy().T, asian_df.to_numpy())
    y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
    (np.dot(np.linalg.inv(XTX), asian_df.to_numpy().T) * y).sum()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
