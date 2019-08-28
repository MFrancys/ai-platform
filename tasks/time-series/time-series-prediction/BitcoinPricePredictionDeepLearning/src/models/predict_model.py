import pandas as pd
import numpy as np
from src.data.processing_data import scale
from src.data.processing_data import invert_scale
from src.data.processing_data import inverse_difference
from src.models.train_model import fit_lstm
import talos as ta

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
        experiment_name="bit",
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
