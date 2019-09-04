### Import basic packages
import os # This module provides a portable way of using operating system dependent functionality.
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime as dt
from datetime import timedelta
import numpy as np # linear algebra
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.feature_extraction import MinimalFCParameters

### Import packages for dataset visualization. Lets use ploty to create dynamy graph.
import plotly as py
from plotly import graph_objs as go
from plotly.offline import plot, iplot, init_notebook_mode
from matplotlib import pyplot
import matplotlib.pyplot as plt


### Import packages for time series analysis
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

### Import modules created to processing and graph data Bitcoin 
from src.timeserie_graphs import ts_graph_line
from src.timeserie_graphs import violin_plot
from src.timeserie_graphs import graph_additive_decomposition
from src.timeserie_graphs import graph_outliers
from src.timeserie_graphs import graph_correlation_features
from src.auxiliary_functions import identify_outliers
from src.auxiliary_functions import test_stationary
from src.build_features import extraction_features_ts

import mlflow

if __name__ == "__main__":
    ###Import and load time series Bitcoin by hour 
    TARGET = "Close"
    ohlc_dict = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume_(Currency)": "sum"
    }
    ROOT_PATH = os.getcwd()
    file_bitcoin =  os.path.join(ROOT_PATH, "data\\ts_bitcoin_2019-01-01_2019-03-13.csv")
    df_hourly_bitcoin = pd.read_csv(file_bitcoin)

    # start mlflow run
    with mlflow.start_run():        
        print("Variable Types \n",df_hourly_bitcoin, "\n")
        print("Bitcoin Time Series Head \n", df_hourly_bitcoin.head(), "\n")
        print("Bitcoin Time Series Tail \n", df_hourly_bitcoin.tail(), "\n")
        
        time_resample = "1D" # End time unit in which events will be predicted

        ###Resample the original dataset close price Bitcoin for hours to events per day, 
        ###in order to obtain the close price Bitcoin presented per day.
        df_daily = df_hourly_bitcoin. \
        set_index(pd.to_datetime(df_hourly_bitcoin["Timestamp"])). \
        resample(rule=time_resample, how=ohlc_dict)
        
        ###Graph Bitcoin TimeSeries
        title=f'BITCOIN TIMESERIE BY DAY'
        fig_graph_line = ts_graph_line(df_daily, TARGET, title)
    #   py.offline.iplot(fig_graph_line)
        
        fig_graph_line_path = os.path.join(ROOT_PATH, "images//bitcointimeseriebyday.png")
        fig_graph_line.write_image(fig_graph_line_path)
        mlflow.log_artifact(fig_graph_line_path) 
            
        ###Graph Additive Decomposition
        graph_additive = graph_additive_decomposition(df_daily, TARGET) #Graph Additive Decomposition
        graph_additive.plot()
        
        ###Normalized Histogram
        x = df_daily[TARGET]
        fig_graph_histogram = go.Figure(data=[go.Histogram(x=x, histnorm='probability')])
        # Edit the layout
        fig_graph_histogram.update_layout(title="Normalized Histogram of Close Price Bitcoin")
    #  fig_graph_histogram.show()
        
        fig_graph_histogram_path = os.path.join(ROOT_PATH, "images//bitcointimeseriebyday_normalizedhistogram.png")
        fig_graph_histogram.write_image(fig_graph_histogram_path)
        mlflow.log_artifact(fig_graph_histogram_path)
        
        ###Graph violin plot by month, weeks and daily of week
        df_daily = df_daily.assign(
            dayofweek=df_daily.index.dayofweek,
            month=df_daily.index.month,
            week=df_daily.index.week,
        )

        time = "month"
        fig_violin_month = violin_plot(df_daily, TARGET, time)
    # fig_violin_month.show()
        
        fig_violin_month_path = os.path.join(ROOT_PATH, "images//bitcointimeseriebyday_violin_month.png")
        fig_violin_month.write_image(fig_violin_month_path)
        mlflow.log_artifact(fig_violin_month_path)
        
        time = "dayofweek"
        fig_violin_dayofweek = violin_plot(df_daily, TARGET, time)
    #    fig_violin_dayofweek.show()
        
        fig_violin_dayofweek_path = os.path.join(ROOT_PATH, "images//bitcointimeseriebyday_violin_dayofweek.png")
        fig_violin_dayofweek.write_image(fig_violin_dayofweek_path)
        mlflow.log_artifact(fig_violin_dayofweek_path)
        
        time = "week"
        fig_violin_week = violin_plot(df_daily, TARGET, time)
    #   fig_violin_week.show()
        
        fig_violin_week_path = os.path.join(ROOT_PATH, "images//bitcointimeseriebyday_violin_week.png")
        fig_violin_week.write_image(fig_violin_week_path)
        mlflow.log_artifact(fig_violin_dayofweek_path)
        
        ###Identify outliers in dataset daily
        df_daily["day_outlier"] = identify_outliers(df_daily, TARGET)    
        fig_graph_outlier = graph_outliers(df_daily, TARGET)
    #    py.offline.iplot(fig_graph_outlier)
            
        fig_graph_outlier_path = os.path.join(ROOT_PATH, "images//bitcointimeseriebyday_outlier.png")
        fig_graph_outlier.write_image(fig_graph_outlier_path)
        mlflow.log_artifact(fig_graph_outlier_path)
        
        # Calculate ACF and PACF upto 50 lags and Draw Plot
        fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
        plot_acf(df_daily[TARGET], lags=50, ax=axes[0])
        plot_pacf(df_daily[TARGET], lags=50, ax=axes[1])
        fig_path = os.path.join(ROOT_PATH, "images//acf_pacf.png")
        fig.savefig(fig_path)
        mlflow.log_artifact(fig_path)
        
        ###Test stationary with ADF Test
        df_daily[f"{TARGET}_diff"], result = test_stationary(df_daily, TARGET)

        mlflow.log_metric("ADF p-value", result)
        
        #Graph Diff Bitcoin TimeSeries
        title=f'BITCOIN STATIONARY TIMESERIE BY DAY'
        TARGET = f"{TARGET}_diff"
        fig_graph_stationary = ts_graph_line(df_daily, TARGET, title)
    #  py.offline.iplot(fig_graph_line)
        
        fig_graph_stationary_path = os.path.join(ROOT_PATH, "images//bitcointimeseriebyday_stationary.png")
        fig_graph_stationary.write_image(fig_graph_stationary_path)
        mlflow.log_artifact(fig_graph_stationary_path)

        ###Feature Extration
        TARGET = "Close_diff"
        df_features = df_daily.reset_index()
        window = 7
        list_features = ["Close_diff", "Timestamp"]
        settings = MinimalFCParameters()

        df_features_bitcoin = extraction_features_ts(df_features, window, settings, TARGET, list_features)
        df_features_bitcoin.to_csv(os.path.join(ROOT_PATH, "data\\ts_features_daily_bitcoin_2019-01-01_2019-03-13.csv"))

        lag = "lag_1"
        fig_correlation_features = graph_correlation_features(df_features_bitcoin, lag)
        #fig_correlation_features.show()

        fig_correlation_features_path = os.path.join(ROOT_PATH, "images//bitcointimeseriebyday_featureslag1.png")
        fig_correlation_features.write_image(fig_correlation_features_path)
        mlflow.log_artifact(fig_correlation_features_path)

        lag = "lag_2"
        fig_correlation_features = graph_correlation_features(df_features_bitcoin, lag)
        #fig_correlation_features.show()

        fig_correlation_features_path = os.path.join(ROOT_PATH, "images//bitcointimeseriebyday_featureslag2.png")
        fig_correlation_features.write_image(fig_correlation_features_path)
        mlflow.log_artifact(fig_correlation_features_path)
