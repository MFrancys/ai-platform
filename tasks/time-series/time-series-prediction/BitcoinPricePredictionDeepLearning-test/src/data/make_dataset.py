# -*- coding: utf-8 -*-
###Import package used to get features timeseries
from statsmodels.tsa.stattools import adfuller, kpss

def test_stationary(df, TARGET):
  #Check Stationary wit ADF Test
  result = adfuller(df[TARGET], autolag='AIC')
  print(f'ADF Statistic: {result[0]}')
  print(f'p-value: {result[1]}')
  for key, value in result[4].items():
      print('Critial Values:')
      print(f'   {key}, {value}')
  if result[1] > 0.01:
    return df[TARGET].diff(), result[1]
  else:
    return df[TARGET], result[1]

def inverse_difference(history, yhat, interval=1):
    # invert differenced value
	return yhat + history[-interval]


if __name__ == '__main__':
    main()
