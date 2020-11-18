import re 
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict

from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def sort_nicely( l ): 
  """ Sort the given list in the way that humans expect. 
  """ 
  convert = lambda text: int(text) if text.isdigit() else text 
  alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
  l.sort( key=alphanum_key ) 


def copy_and_modify_dict(_dict: dict, update_dict: dict):
    copy_dict = _dict.copy()
    copy_dict.update(update_dict)
    return copy_dict


# ML models helper functions.
# ================================================================================


def predict_next_window_recursive(model, input_data, future_window=14):
    predictions = []
    x_batch = [val for val in input_data]
    for _ in range(future_window):
        # Predict the value and append it to predictions list.
        pred_val = model.predict(np.asarray([x_batch]))
        predictions.append(pred_val[0])

        # Modify the x_batch: pop the first value and add pred_val to last index.
        x_batch.append(pred_val[0])
        x_batch.pop(0)

    return np.asarray(predictions)


def predict_next_window_direct(models, input_data):
    predictions = []
    for model in models:
        predictions.append(model.predict(np.expand_dims(input_data, 0)))

    return np.asarray(predictions)


# Plot functions.
# ================================================================================
def plot_serie_with_next_window_prediction(model,
                                           train_df,
                                           test_df,
                                           scaler,
                                           w_size,
                                           f_steps,
                                           col_idx=33,
                                           recursive=False,
                                           title='',
                                           figsize=(18, 6)):
    """This function assumes you're using a DataFrame with unscaled values"""
    fig = plt.figure(figsize=figsize)
    plt.title(title)

    # Inverse transform the train values and plot them.
    train_x_values = train_df.index
    train_y_values = train_df.values[:, col_idx]
    plt.plot(train_x_values,
             train_y_values,
             c='black',
             label='Casos de ajuste')

    # Split the test df in three parts: one for the window values used for
    # predicting the last values from the test df, one for the last possible
    # predicted values from the test df and one df for the remaining values.
    rem_df = test_df.iloc[:-(w_size + f_steps)]
    rem_x_values = rem_df.index
    rem_y_values = rem_df.values[:, col_idx]
    plt.plot(rem_x_values, rem_y_values, c='blue', label='Casos de prueba')

    w_df = test_df.iloc[-(w_size + f_steps):-f_steps]
    w_x_values = w_df.index
    w_y_values = w_df.values[:, col_idx]
    plt.plot(w_x_values,
             w_y_values,
             c='orange',
             label='Ventana utilizada para predecir')

    last_df = test_df.iloc[-f_steps:]
    last_x_values = last_df.index
    last_y_values = last_df.values[:, col_idx]
    plt.plot(last_x_values,
             last_y_values,
             c='green',
             label='Ultimos valores reales')

    if isinstance(model, tf.keras.Model):
        if scaler.var_.shape[0] > 1:
            x_values = scaler.transform(w_df.values)
            pred_values = model.predict(np.expand_dims(x_values, axis=0))
            tem_last = np.zeros((last_df.shape))
            tem_last[:,col_idx] = pred_values.ravel()       
            #last_df.loc[:,'BUCARAMANGA'] = pred_values.ravel()
            pred_values = scaler.inverse_transform(tem_last)
            pred_values = pred_values[:,col_idx]
        else:
            x_values = scaler.transform(np.expand_dims(w_df.values[:, col_idx], axis=-1))
            pred_values = model.predict(np.expand_dims(x_values.reshape(-1,1), axis=0))
            pred_values = scaler.inverse_transform(pred_values)
    else:
	# Predict the values after the last window.
        x_values = scaler.transform(np.expand_dims(w_df.values[:, col_idx], axis=-1))
        if recursive:
            pred_values = predict_next_window_recursive(model,
                                                        x_values[..., 0],
                                                        future_window=f_steps)
        else:
            pred_values = predict_next_window_direct(model, x_values[..., 0])

        pred_values = scaler.inverse_transform(pred_values)
    plt.plot(last_x_values,
             pred_values.ravel(),
             c='red',
             label='Valores predichos',
             marker='o',
             markersize=4)

    plt.legend()


def predict_and_plot_next_window_from_date(model,
                                           df,
                                           scaler,
                                           start_date,
                                           w_size,
                                           f_steps,
                                           col_idx=33,
                                           recursive=False,
                                           title='',
                                           figsize=(18, 6)):

    fig = plt.figure(figsize=figsize)
    plt.title(title)

    w_df = df.loc[df.index >= start_date]
    w_df = w_df.iloc[:w_size]
    w_x_values = w_df.index
    w_y_values = w_df.values[:, col_idx]
    plt.plot(w_x_values,
             w_y_values,
             c='blue',
             label='Ventana utilizada para predecir')

    f_df = df.loc[df.index >= start_date]
    f_df = f_df.iloc[w_size:(w_size + f_steps)]
    f_x_values = f_df.index
    f_y_values = f_df.values[:, col_idx]
    plt.plot(f_x_values,
             f_y_values,
             c='orange',
             label='Ultimos valores reales')

    # Predict the values after the last window.
    x_values = scaler.transform(
        np.expand_dims(w_df.values[:, col_idx], axis=-1))

    if isinstance(model, tf.keras.Model):
        pred_values = model.predict(np.expand_dims(x_values.reshape(-1,1), axis=0))
    else:
        if recursive:
            pred_values = predict_next_window_recursive(model,
                                                        x_values[..., 0],
                                                        future_window=f_steps)
        else:
            pred_values = predict_next_window_direct(model, x_values[..., 0])

    pred_values = scaler.inverse_transform(pred_values)
    plt.plot(f_x_values,
             pred_values.ravel(),
             c='red',
             label='Valores predichos',
             marker='o',
             markersize=4)

    plt.legend()


# Functions to analyze the series | Credits: github.com/madagra/
class TargetTransformer:
    """
    Perform some transformation on the time series
    data in order to make the model more performant and
    avoid non-stationary effects.
    """
    def __init__(self, log=False, detrend=False, diff=False):

        self.trf_log = log
        self.trf_detrend = detrend
        self.trend = pd.Series(dtype=np.float64)

    def transform(self, index, values):
        """
        Perform log transformation to the target time series

        :param index: the index for the resulting series
        :param values: the values of the initial series

        Return:
            transformed pd.Series
        """
        res = pd.Series(index=index, data=values)

        if self.trf_detrend:
            self.trend = TargetTransformer.get_trend(res) - np.mean(res.values)
            res = res.subtract(self.trend)

        if self.trf_log:
            res = pd.Series(index=index, data=np.log(res.values))

        return res

    def inverse(self, index, values):
        """
        Go back to the original time series values

        :param index: the index for the resulting series
        :param values: the values of series to be transformed back

        Return:
            inverse transformed pd.Series
        """
        res = pd.Series(index=index, data=values)

        if self.trf_log:
            res = pd.Series(index=index, data=np.exp(values))
        try:
            if self.trf_detrend:
                assert len(res.index) == len(self.trend.index)
                res = res + self.trend

        except AssertionError:
            print("Use a different transformer for each target to transform")

        return res

    @staticmethod
    def get_trend(data):
        """
        Get the linear trend on the data which makes the time
        series not stationary
        """
        n = len(data.index)
        X = np.reshape(np.arange(0, n), (n, 1))
        y = np.array(data)
        model = LinearRegression()
        model.fit(X, y)
        trend = model.predict(X)
        return pd.Series(index=data.index, data=trend)


def ts_analysis_plots(data, n_lags=100):
    def plot_cf(ax, fn, data, n_lags):
        """
        Plot autocorrelacción
        """
        fn(data, ax=ax, lags=n_lags, color="#0504aa")
        for i in range(1, 5):
            ax.axvline(x=24 * i, ymin=0.0, ymax=1.0, color='grey', ls="--")

    # AD Fuller test and linear trend of the time series
    trend = TargetTransformer.get_trend(data)
    adf = adfuller(data)

    fig, axs = plt.subplots(2, 2, figsize=(25, 12))
    axs = axs.flat

    # original time series
    axs[0].plot(data, color='#0504aa')
    axs[0].plot(trend, color="red")
    axs[0].set(
        xlabel="Fecha",
        ylabel="Vslores",
        title=f"Casos COVID-19 en Bucaramanga (ADF p-value: {round(adf[1], 6)})"
    )

    # histogram of value distribution
    axs[1].hist(data, bins=20, width=3, color='#0504aa', alpha=0.7)
    axs[1].set(xlabel="Casos",
               ylabel="Frecuencia",
               title="Distribución de casos COVID-19")

    # autocorrelation function
    plot_cf(axs[2], plot_acf, data, n_lags)
    axs[2].set(xlabel="lag", ylabel="Valor ACF")

    # partial autocorrelation function
    plot_cf(axs[3], plot_pacf, data, n_lags)
    axs[3].set(xlabel="lag", ylabel="Valor PACF")

    plt.show()

# for preproccesing data
def preproccesing_data(data, data_colombia):
    # all countries
    timeline = sorted(np.unique(data.date.astype('datetime64[ns]')))
    data_selected = data.loc[(~data.iso_code.isna()) & (data.iso_code != 'OWID_WRL'), [
        'iso_code', 'continent', 'location', 'date', 'new_cases',
        'new_cases_smoothed', 'new_deaths', 'new_deaths_smoothed']]

    country_freq = data_selected.groupby(['location']).agg({'date': 'count'}).rename(columns={'date': 'num_records'})
    records, counts = np.unique(country_freq.num_records.values, return_counts=True)
    result_list = list(zip(records[np.argsort(counts)][::-1],counts[np.argsort(counts)][::-1]))
    #print('Max group of records {}'.format(max(counts)))
    #print('Order by num of records {}'.format(result_list))
    max_value = result_list[0][0]
    print('Max value in days {}'.format(max_value))
    
    # build the structure and apply padding based on max value of days
    data_to_use = defaultdict(list)
    # use data tem to less countries
    for index, row in data_selected.iterrows():
        data_to_use[row.location].append(row.new_cases)

    final_dict = dict()
    for country in data_to_use.keys():
        diff = max_value - len(data_to_use[country])
        final_dict[country] = [0.0] * diff + data_to_use[country]

    del data_to_use
    data_to_use = pd.DataFrame(final_dict)
    data_to_use['date'] = timeline
    data_to_use.set_index('date', inplace=True)
    data_to_use.fillna(axis=0, method='backfill', inplace=True)

    ##### Data Colombia
    # how many cases are there by dep
    df_freq_dep = data_colombia.groupby(['Nombre departamento']).count()['Fecha de inicio de síntomas'].to_frame().rename(
                                    columns={'Fecha de inicio de síntomas': 'Casos'})
    # how many cases are there by country
    df_freq = data_colombia.groupby(['Nombre municipio']).count()['Fecha de inicio de síntomas'].to_frame().rename(
                                    columns={'Fecha de inicio de síntomas': 'Casos'})
    # group dep and city data by 'Fecha de inicio de síntomas'
    data_colombia_by_city = data_colombia.groupby(
        ['Nombre municipio', 'Fecha de inicio de síntomas']).agg({
            'Fecha de inicio de síntomas':
            'count'
        }).rename(columns={'Fecha de inicio de síntomas': 'cases'})

    data_colombia_by_dep = data_colombia.groupby(
        ['Nombre departamento', 'Fecha de inicio de síntomas']).agg({
            'Fecha de inicio de síntomas':
            'count'
        }).rename(columns={'Fecha de inicio de síntomas': 'cases'})

    # unstack and joins the dataframes
    data_colombia_by_city = data_colombia_by_city.cases.unstack().T.fillna(axis=0, method='backfill', inplace=False)
    data_colombia_by_dep = data_colombia_by_dep.cases.unstack().T.fillna(axis=0, method='backfill', inplace=False)
    # cities and dep with the same name - dont take into account for both dataframes
    to_include = list(set(data_colombia_by_dep.keys()).difference(set(data_colombia_by_city.keys())))
    # join
    data_colombia_main = data_colombia_by_city.join(data_colombia_by_dep[to_include])
    # fix the index
    #data_colombia_main['date'] = data_colombia_main.index.astype('datetime64[ns]')
    data_colombia_main['date'] = pd.to_datetime(data_colombia_main.index,
                                            infer_datetime_format=False,
                                            format='%d/%m/%Y %H:%M:%S')
    data_colombia_main.set_index('date', inplace=True)
    # sincronize the sequences
    df_join = data_to_use.join(data_colombia_main)
    df_join.fillna(axis=0, method='backfill', inplace=True)
 	
    return df_join