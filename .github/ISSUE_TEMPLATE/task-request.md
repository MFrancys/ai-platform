---
name: BitcoinPricePrediction
about: Bitcoin Price Closing Daily Prediction with Deep Learning
title: 'PREDICTION OF DAILY CLOSING PRICES OF BITCOIN WITH RECURRENT NEURAL NETWORKS (with LSTM) AND HYPERPARAMETER OPTIMIZATION WITH KERAS MODELS'
labels: task
assignees: ''

---

# Goal(s)

- Predict Daily Closing Price Bitcoin wiht Deep Learnning and Hyperparameter

# Input(s)

- Bitcoin daily Prices

# Output(s)

- Predictions of Daily Closing Price Bitcoin
- Best Model LSTM

# Objective Function(s)

- extraction_features_ts
- get_predictions_lstm
- graph_predictions

---
name: AnalysisTweets
about: AnalysisTweets get the tweets posted by a user and apply NLP of them
title: 'Analysis of a user's tweets with NLP'
labels: task
assignees: ''

---

# Goal(s)

- TWEETS ANALYSIS PUBLISHED BY DONALD TRUMP WITH NATURAL PROCESSING LANGUAGE(NLP)

# Input(s)

- Twitter Credentials
- User ScreenName

# Output(s)

- Account information
- Number of tweets
- Number or retweets
- Number of positive tweets
- Number of negative tweets
- Word Cloud
- Most used words in the twitters
- Dominant topic in each tweets
- Topics distribution across tweets
- Topic-keyword Matrix
- Tweets cluster by similarity of Topics

# Objective Function(s)

- graph_analysis_twitter
- classes_wordcloud
- graph_count_words
- get_best_lda
- show_topics
- KMeans(sklearn)
- TruncatedSVD(sklearn)

---
name: BitcoinTimeSeriesAnalysis
about: Bitcoin Daily Closing Price Analysis and automatic feature extration
title: 'UNIVARIATE_TIME_SERIES_ANALYSIS_AND_AUTOMATIC_FEATURE_EXTRATION - BITCOIN'
labels: task
assignees: ''

---

# Goal(s)

- Identify patterns in the time series daily Bitcoin and automatic feature extration

# Input(s)

- Bitcoin hourly Prices

# Output(s)

- Graph of the time series of daily close price Bitcoin
- Graph Additive time series Bitcoin Prices
- Normalized Histogram
- Graph violin plot by month, weeks and daily of week
- Identify outliers
- Test Stationary and non-stationary of the time series
- Compute autocorrelation and partial autocorrelation functions of the univariate time serie
- Extration daily features
- Compute correlation of features

# Objective Function(s)

- processing_dataset
- extraction_features_ts
