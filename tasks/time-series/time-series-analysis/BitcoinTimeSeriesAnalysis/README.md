BitcoinTimeSeriesAnalysis
==============================

Version: 1.0a0

Platform: Windows

Summary: Bitcoin Daily Closing Price Analysis

Keywords: bitcoin UnivarianteTimeSeries Visualization ExtractionFeatures

Bitcoin Price Closing Daily Analysis

Installation:
Create a virtual environment and activate it.
```bash
1) conda env create -f conda.yaml
2) conda activate timeseries_myenv
```

Usage:
```bash
1) python timeseriesanalysis.py
2) mlflow ui
```

Description:
This is a project made with MLflow to make a time series analysis on the daily price of bitcoin. Feature extraction is done automatically using a package called tsfresh.

The analysis has to follow the next steps:
    1) Visualizing of the time series of daily close price Bitcoin
    2) Identify patterns in the time series daily Bitcoin
        - A time series may be split into the following components: Base Level + Trend + Seasonality + Error.
        Depending on the nature of the trend and seasonality, a time series can be modeled as an additive or multiplicative,
        where in each observation in the series can be expressed as either a sum or a product of the components:
        Additive time series:
            Value = Base Level + Trend + Seasonality + Error
        - Normalized Histogram
        - Graph violin plot by month, weeks and daily of week 
    3) Identify outliers in dataset daily. The method selected to detect outliers by days is IQR Score.
    4) Test Stationary and non-stationary of the time series
    5) Compute autocorrelation and partial autocorrelation functions of the univariate time serie
    6) Extration daily features
    7) Compute correlation of features

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Bitcoin Data taken from Kaggle https://www.kaggle.com/mczielinski/bitcoin-historical-data
    │
    ├── conda.yaml         <- The requirements for reproducing the analysis environment
    │
    ├── src                <- Source code for use in this project.
    │   │── auxiliary_functions.py
    │   │── build_features.py
    │   └── timeserie_graphs.py
    │
    ├── MLproject          <- The requirements for reproducing the analysis environment
    │
    ├── images             <- Folder where graphics pngs are saved
    │
    ├── outputs            <- Folder where values of the analysis are saved in a csv file
    │
    └── timeserie_graphs.py   <- Source code principal to make daily closing price bitcoin analysis

--------

