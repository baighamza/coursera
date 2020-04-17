# Building a Regression Model for a Financial Dataset

In this notebook, you will build a simple linear regression model to predict the closing AAPL stock price. The lab objectives are:
* Pull data from BigQuery into a Pandas dataframe
* Use Matplotlib to visualize data
* Use Scikit-Learn to build a regression model


```bash
%%bash

bq mk -d ai4f
bq load --autodetect --source_format=CSV ai4f.AAPL10Y gs://cloud-training/ai4f/AAPL10Y.csv
```


```python
%matplotlib inline

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

plt.rc('figure', figsize=(12, 8.0))
```

## Pull Data from BigQuery

In this section we'll use a magic function to query a BigQuery table and then store the output in a Pandas dataframe. A magic function is just an alias to perform a system command. To see documentation on the "bigquery" magic function execute the following cell:


```python
%%bigquery?
```


    [0;31mDocstring:[0m
    ::
    
      %_cell_magic [--destination_table DESTINATION_TABLE] [--project PROJECT]
                       [--max_results MAX_RESULTS]
                       [--maximum_bytes_billed MAXIMUM_BYTES_BILLED] [--dry_run]
                       [--use_legacy_sql] [--use_bqstorage_api] [--verbose]
                       [--params PARAMS [PARAMS ...]]
                       [destination_var]
    
    Underlying function for bigquery cell magic
    
    Note:
        This function contains the underlying logic for the 'bigquery' cell
        magic. This function is not meant to be called directly.
    
    Args:
        line (str): "%%bigquery" followed by arguments as required
        query (str): SQL query to run
    
    Returns:
        pandas.DataFrame: the query results.
    
    positional arguments:
      destination_var       If provided, save the output to this variable instead
                            of displaying it.
    
    optional arguments:
      --destination_table DESTINATION_TABLE
                            If provided, save the output of the query to a new
                            BigQuery table. Variable should be in a format
                            <dataset_id>.<table_id>. If table does not exists, it
                            will be created. If table already exists, its data
                            will be overwritten.
      --project PROJECT     Project to use for executing this query. Defaults to
                            the context project.
      --max_results MAX_RESULTS
                            Maximum number of rows in dataframe returned from
                            executing the query.Defaults to returning all rows.
      --maximum_bytes_billed MAXIMUM_BYTES_BILLED
                            maximum_bytes_billed to use for executing this query.
                            Defaults to the context
                            default_query_job_config.maximum_bytes_billed.
      --dry_run             Sets query to be a dry run to estimate costs. Defaults
                            to executing the query instead of dry run if this
                            argument is not used.
      --use_legacy_sql      Sets query to use Legacy SQL instead of Standard SQL.
                            Defaults to Standard SQL if this argument is not used.
      --use_bqstorage_api   [Beta] Use the BigQuery Storage API to download large
                            query results. To use this option, install the google-
                            cloud-bigquery-storage and fastavro packages, and
                            enable the BigQuery Storage API.
      --verbose             If set, print verbose output, including the query job
                            ID and the amount of time for the query to finish. By
                            default, this information will be displayed as the
                            query runs, but will be cleared after the query is
                            finished.
      --params <PARAMS [PARAMS ...]>
                            Parameters to format the query string. If present, the
                            --params flag should be followed by a string
                            representation of a dictionary in the format
                            {'param_name': 'param_value'} (ex. {"num": 17}), or a
                            reference to a dictionary in the same format. The
                            dictionary reference can be made by including a '$'
                            before the variable name (ex. $my_dict_var).
    [0;31mFile:[0m      /opt/conda/lib/python3.7/site-packages/google/cloud/bigquery/magics.py



The query below selects everything you'll need to build a regression model to predict the closing price of AAPL stock. The model will be very simple for the purposes of demonstrating BQML functionality. The only features you'll use as input into the model are the previous day's closing price and a three day trend value. The trend value can only take on two values, either -1 or +1. If the AAPL stock price has increased over any two of the previous three days then the trend will be +1. Otherwise, the trend value will be -1.

Note, the features you'll need can be generated from the raw table `ai4f.AAPL10Y` using Pandas functions. However, it's better to take advantage of the serverless-ness of BigQuery to do the data pre-processing rather than applying the necessary transformations locally.  


```python

```


```python

```


```python
%%bigquery df
WITH
  raw AS (
  SELECT
    date,
    close,
    LAG(close, 1) OVER(ORDER BY date) AS min_1_close,
    LAG(close, 2) OVER(ORDER BY date) AS min_2_close,
    LAG(close, 3) OVER(ORDER BY date) AS min_3_close,
    LAG(close, 4) OVER(ORDER BY date) AS min_4_close
  FROM
    `ai4f.AAPL10Y`
  ORDER BY
    date DESC ),
  raw_plus_trend AS (
  SELECT
    date,
    close,
    min_1_close,
    IF (min_1_close - min_2_close > 0, 1, -1) AS min_1_trend,
    IF (min_2_close - min_3_close > 0, 1, -1) AS min_2_trend,
    IF (min_3_close - min_4_close > 0, 1, -1) AS min_3_trend
  FROM
    raw ),
  train_data AS (
  SELECT
    date,
    close,
    min_1_close AS day_prev_close,
    IF (min_1_trend + min_2_trend + min_3_trend > 0, 1, -1) AS trend_3_day
  FROM
    raw_plus_trend
  ORDER BY
    date ASC )
SELECT
  *
FROM
  train_data
```

View the first five rows of the query's output. Note that the object `df` containing the query output is a Pandas Dataframe.


```python
print(type(df))
df.dropna(inplace=True)
df.head()
```

## Visualize data

The simplest plot you can make is to show the closing stock price as a time series. Pandas DataFrames have built in plotting funtionality based on Matplotlib. 


```python
df.plot(x='date', y='close');
```

You can also embed the `trend_3_day` variable into the time series above. 


```python
start_date = '2018-06-01'
end_date = '2018-07-31'

plt.plot(
    'date', 'close', 'k--',
    data = (
        df.loc[pd.to_datetime(df.date).between(start_date, end_date)]
    )
)

plt.scatter(
    'date', 'close', color='b', label='pos trend', 
    data = (
        df.loc[df.trend_3_day == 1 & pd.to_datetime(df.date).between(start_date, end_date)]
    )
)

plt.scatter(
    'date', 'close', color='r', label='neg trend',
    data = (
        df.loc[(df.trend_3_day == -1) & pd.to_datetime(df.date).between(start_date, end_date)]
    )
)

plt.legend()
plt.xticks(rotation = 90);
```


```python
df.shape
```

## Build a Regression Model in Scikit-Learn

In this section you'll train a linear regression model to predict AAPL closing prices when given the previous day's closing price `day_prev_close` and the three day trend `trend_3_day`. A training set and test set are created by sequentially splitting the data after 2000 rows. 


```python
features = ['day_prev_close', 'trend_3_day']
target = 'close'

X_train, X_test = df.loc[:2000, features], df.loc[2000:, features]
y_train, y_test = df.loc[:2000, target], df.loc[2000:, target]
```
# Create linear regression object. Don't include an intercept,

regr = linear_model.LinearRegression(fit_intercept=False)

```python
# Train the model using the training set
regr.fit(X_train, y_train)
```


```python
# Make predictions using the testing set
testing_predict = regr.predict(X_test)
```


```python
# Print the root mean squared error of your predictions
print("Root mean squared error: {0:.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
```


```python
# Print the variance score (1 is perfect prediction)
print("Variance score: {0:.2f}".format(r2_score(y_test, y_pred)))
```


```python
# Plot the predicted values against their corresponding true values
plt.scatter(y_test, y_pred)
plt.plot([140, 240], [140, 240], 'r--', label='perfect fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend();
```

The model's predictions are more or less in line with the truth. However, the utility of the model depends on the business context (i.e. you won't be making any money with this model). It's fair to question whether the variable `trend_3_day` even adds to the performance of the model:


```python
print('Root Mean Squared Error: {0:.2f}'.format(np.sqrt(mean_squared_error(y_test, X_test.day_prev_close))))
```

Indeed, the RMSE is actually lower if we simply use the previous day's closing value as a prediction! Does increasing the number of days included in the trend improve the model? Feel free to create new features and attempt to improve model performance!
