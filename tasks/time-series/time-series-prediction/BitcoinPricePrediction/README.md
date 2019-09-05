BitcoinPricePredictions
==============================

Version: 1.0a0

Platform: Windows

Summary: Bitcoin Daily Closing Price Predictions

Keywords: bitcoin UnivarianteTimeSeries RecurrentNeuralNetworks ExtractionFeatures Predictions HyperparameterOptimization

Bitcoin Price Closing Daily Analysis

Installation:
Create a virtual environment and activate it.
```bash
1) conda env create -f conda.yaml
2) conda activate tf_myenv
```

Usage:
```bash
1) python predict_model.py
2) mlflow ui
```

Description:
This is a project made with MLflow to make daily predictions of the closing price of bitcoin with the implementation of recurrent neural networks(LSTM) with keras and with tesorflow as backend. Also, hyperparameter optimization was applied with the package called Talos, where the hyperparameters set were number of neurons, dropout, optimizer and last activation.
Feature extraction is done automatically using a package called tsfresh.

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Bitcoin Data taken from Kaggle https://www.kaggle.com/mczielinski/bitcoin-historical-data
    │
    ├── conda.yaml         <- The requirements for reproducing the analysis environment
    │
    ├── src                <- Source code for use in this project.
    │   │── processing_features.py
    │   │── build_features.py
    │   └── train_model.py
    │
    ├── MLproject          
    │
    ├── outputs            <- Folder where outputs(values, graphs) are saved 
    │
    └── preditc_model.py   <- Source code principal to make daily closing price bitcoin predictions
--------

