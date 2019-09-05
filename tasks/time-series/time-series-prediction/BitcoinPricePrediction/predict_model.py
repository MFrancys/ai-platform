### Import basic packages
import os
import pandas as pd
import numpy as np

### Import Own Packages to processing features, build features and train model
from src.processing_features import scale
from src.processing_features import invert_scale
from src.processing_features import inverse_difference
from src.processing_features import test_stationary
from src.build_features import extraction_features_ts
from src.train_model import fit_lstm

### Import packages for dataset visualization. Lets use ploty to create dynamy graph.
from src.visualize import graph_predictions

### Import Hyperparameter Optimization Packages
import talos as ta


from tsfresh.feature_extraction import MinimalFCParameters

import mlflow

def get_predictions_lstm(df_daily_features, n_predictions, max_time, TARGET, p, adf_test, raw_values):

  ### Split data into train and test-set
  supervised_values = df_daily_features.values
  train, test = supervised_values[0:-n_predictions], supervised_values[-n_predictions:]

  predictions = list()
  list_expected = list()

  ### Walk-forward validation on the test data
  for i in range(len(test)):

      # transform the scale of the data
      train = train[-max_time:,:]
      scaler, train_scaled, test_scaled = scale(train, test)

      x_train, y_train = train_scaled[:, :-1], train_scaled[:, -1:]
      x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])

      ### Training Neuronal Network
      h = ta.Scan(x_train, y_train,
        params=p,
        model=fit_lstm,
        #                  dataset_name='bitcoin',
        #                  experiment_no='1',
        seed=123,
#        grid_downsample=0.1,
        experiment_name="models",
#        reduction_threshold=0.1,
        round_limit = 1
      )

      ### Get best model
      best_model = h.best_model(metric='loss', asc=True)

      ###make predictions
      x_test, y_test = test_scaled[i:i+1, 0:-1], test_scaled[i:i+1, -1]
      x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
      yhat = best_model.predict(x_test)[0][0]

      # invert scaling
      yhat = invert_scale(scaler, x_test, yhat)

      # invert differencing
      if adf_test > 0.01:
          yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)

      # store forecast
      predictions.append(yhat)

      # store real value
      expected = raw_values[-(len(test) -(i))]
      list_expected.append(expected)

      ###add row of prediction to train
      train = np.append(train, [test[i]], axis=0)

      print('day={}, Predicted={}, Expected={}'.format(i, yhat, expected))
      print("Second :" + str(len(train)))

  df_predictions = pd.DataFrame(
    index=df_daily_features[-n_predictions:].index,
    data=predictions,
    columns=["predictions"]
  )

  df_predictions["predictions"] = df_predictions.predictions.round(2)
  df_predictions[TARGET] = list_expected

  return df_predictions, best_model

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
    file_bitcoin =  os.path.join(ROOT_PATH, "data\\ts_daily_bitcoin_2019-01-01_2019-03-13.csv")
    df_hourly_bitcoin = pd.read_csv(file_bitcoin)

    time_resample = "1D" # End time unit in which events will be predicted

    ###Resample the original dataset close price Bitcoin for hours to events per day, 
    ###in order to obtain the close price Bitcoin presented per day.
    df_daily_bitcoin = df_hourly_bitcoin. \
      set_index(pd.to_datetime(df_hourly_bitcoin["Timestamp"])). \
      resample(rule=time_resample, how=ohlc_dict)

    TARGET = "Close"
    ###Test for stationary
    df_daily_bitcoin["Close_diff"], adf_test = test_stationary(df_daily_bitcoin, TARGET)
    df_daily_bitcoin = df_daily_bitcoin.dropna(subset=["Close_diff"])

    raw_values = df_daily_bitcoin[TARGET].values

    ###Feature Extration
    TARGET = "Close_diff"
    df_features = df_daily_bitcoin.reset_index()
    window = 7
    list_features = ["Close_diff", "Timestamp"]
    settings = MinimalFCParameters()
    df_features_bitcoin = extraction_features_ts(df_features, window, settings, TARGET, list_features)

    print(df_features_bitcoin.columns)

    # start mlflow run
    with mlflow.start_run():  
      n_predictions = 12
      max_time = 60
      neurons = len(df_features_bitcoin.columns)

      ### Parameters of the neural network to optimize
      p = {
        'lr': (0.1, 10, 10),
        'first_neuron':[neurons, neurons*2, neurons*3],
        'second_neuron':[neurons*3, neurons*4, neurons*5],
        'batch_size': [1],
        'epochs': [500],
        'dropout': (0.40, 0.50, 0.60),
        'optimizer': ["Adam", "Nadam", "SGD"],
        #     'loss': ['categorical_crossentropy'],
        'last_activation': ['relu', "elu"],
        #     'weight_regulizer': [None]
      }

      df_predictions, best_model = get_predictions_lstm(df_features_bitcoin, n_predictions, max_time, TARGET, p, adf_test, raw_values)
      df_predictions = df_predictions. \
        reset_index(). \
        rename(columns={"index": "Timestamp", "Close_diff": "Close"})
      print(df_predictions)

      df_predictions_path = os.path.join(ROOT_PATH, "outputs\\bitcointimeseriebyday_predictions.csv")
      df_predictions.to_csv(df_predictions_path)
      mlflow.log_artifact(df_predictions_path)

      TARGET = "Close"
      fig_predictions = graph_predictions(df_predictions, TARGET)

      fig_predictions_path = os.path.join(ROOT_PATH, "outputs\\bitcointimeseriebyday_predictions.png")
      fig_predictions.write_image(fig_predictions_path)
      mlflow.log_artifact(fig_predictions_path)
