import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler


def choose_shuffled_order(size, seed=123):
    order = np.arange(size)
    np.random.RandomState(seed).shuffle(order)
    return order


def series_transformer(series_df, standarization, scaler=None):

    # Inputar valores faltantes sobre la(s) serie(s).
    series_transformed = series_df.fillna(axis=0,
                                          method='backfill',
                                          inplace=False)

    # No hay entonces ajuste uno.
    if standarization and scaler == None:
        scaler = StandardScaler()
        scaler = scaler.fit(series_transformed.values)
        values = scaler.transform(series_transformed.values)
        series_transformed = pd.DataFrame(data=values,
                                          columns=series_transformed.keys(),
                                          index=series_transformed.index)

    # Estandarizar con uno existente (test).
    elif standarization and scaler != None:
        values = scaler.transform(series_transformed.values)
        series_transformed = pd.DataFrame(data=values,
                                          columns=series_transformed.keys(),
                                          index=series_transformed.index)

    return series_transformed, scaler


def windowed_dataset(series, dset_params, tf_dset=True, direct_dset=False):

    # Extract values from dict.
    batch_size = dset_params['batch_size']
    past_window = dset_params['past_window']
    future_window = dset_params['future_window']
    shuffle_buffer = dset_params['shuffle_buffer']
    col_test_series = dset_params['col_test_series']

    assert past_window > 0
    assert future_window > 0
    shuffle = True if shuffle_buffer > 0 else False

    # convert the series to tensor format
    w_dset = tf.data.Dataset.from_tensor_slices(series)
    w_dset = w_dset.window(past_window + future_window,
                           shift=1,
                           drop_remainder=True)

    # convert each window to numpy vector format
    w_dset = w_dset.flat_map(
        lambda window: window.batch(past_window + future_window))

    # choose t --> t-future_target as a features and the last one as a label
    w_dset = w_dset.map(lambda window: (window[:-future_window, :], window[
        -future_window:, col_test_series]))

    if tf_dset:
        dataset = w_dset.cache().shuffle(shuffle_buffer,
                                         reshuffle_each_iteration=True)
        dataset = w_dset.batch(batch_size).prefetch(
            tf.data.experimental.AUTOTUNE)
    else:
        x_data = []
        dataset = dict()

        # Check if the dset is needed for training a direct ML model.
        if direct_dset and future_window > 1:
            dataset['x'], dataset['y'] = [], []
            for idx in range(future_window):
                y_data = []
                for elem in w_dset.as_numpy_iterator():
                    if idx == 0:
                        x_data.append(elem[0])
                    y_data.append(elem[1][idx])

                if idx == 0:
                    x_data = np.asarray(x_data).reshape(-1, past_window)
                    if shuffle:
                        random_order = choose_shuffled_order(x_data.shape[0])
                        x_data = x_data[random_order]
                
                y_data = np.asarray(y_data).reshape(-1)
                if shuffle: y_data = y_data[random_order]

                dataset['x'].append(x_data)
                dataset['y'].append(y_data)

        else:
            y_data = []
            for elem in w_dset.as_numpy_iterator():
                x_data.append(elem[0])
                y_data.append(elem[1])

            x_data = np.asarray(x_data)
            y_data = np.asarray(y_data)
            
            if shuffle: 
                random_order = choose_shuffled_order(x_data.shape[0])
                x_data = x_data[random_order]
                y_data = y_data[random_order]

            dataset['x'] = x_data.reshape(-1, past_window)
            dataset['y'] = y_data.reshape(-1)

    return dataset