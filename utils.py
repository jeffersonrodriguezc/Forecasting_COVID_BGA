import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def copy_and_modify_dict(_dict:dict, update_dict: dict):
    copy_dict = _dict.copy()
    copy_dict.update(update_dict)
    return copy_dict

# Funciones para analizar las series | Créditos: github.com/madagra/
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