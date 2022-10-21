from keras.layers import Dense, Dropout
from keras.models import Sequential
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

random.seed(10)

def generate_model(in_d):
    """
    Model
    """
    model = Sequential()
    model.add(Dense(16, activation='linear',
                    input_dim=in_d))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(
        Dense(1, activation="linear"))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def add_residuals(synth_data):
    """
    Adds noise to synthetic NEE based on temperature
    """
    synth_data['residual'] = (synth_data.NEE-synth_data.modelled_NEE).values
    assignments, edges = pd.qcut(
        synth_data['TA_F_MDS'], 5, retbins=True, labels=False)
    synth_data['label'] = assignments

    for label in synth_data.label.unique():
        cur = synth_data[(synth_data.label == label) & (
            synth_data.residual.notnull())]
        if len(cur) < 20:
            cur = synth_data[(synth_data.label <= label+1) & (
            synth_data.residual.notnull())]
            if len(cur) < 20:
                cur = synth_data[(synth_data.label <= label+2) & (
                synth_data.residual.notnull())]
                if len(cur) < 20:
                    cur = synth_data[(synth_data.label <= label+3) & (
                synth_data.residual.notnull())]

        res = cur.residual.sample(
            len(synth_data[(synth_data.label == label)]), replace=True).values

        synth_data.loc[(synth_data.label == label),
                           'synth_NEE'] = synth_data.loc[(synth_data.label == label), 'modelled_NEE'] + res

    new = synth_data[['synth_NEE', 'TA_F_MDS', 'TS_F_MDS_1', 'SW_IN_F', 'VPD_F_MDS']].copy()
    new.columns = ['NEE', 'TA_F_MDS', 'TS_F_MDS_1', 'SW_IN_F','VPD_F_MDS']

    return new


def make_synth_data(data, x_cols, y_col):
    """
    Makes a synthetic data set using neural network
    """
    all_train_data = data[(data.NEE.notnull())]
    X = np.asarray(all_train_data[x_cols])
    y = np.asarray(all_train_data[y_col])
    X, y = shuffle(X, y)
    scaler_x = StandardScaler().fit(X)
    X = scaler_x.transform(X)
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))
    y = scaler_y.transform(y.reshape(-1, 1))

    X_all = scaler_x.transform(np.asarray(data[x_cols]))

    model = generate_model(len(x_cols))
    history = model.fit(X, y, epochs=100, batch_size=32)

    pred_all_y = model.predict(X_all)
    pred_all_y = scaler_y.inverse_transform(pred_all_y)

    data['modelled_NEE'] = pred_all_y
    
    winter = data[data.index.month.isin([1, 2, 3, 4, 11, 12])].copy()
    summer = data[data.index.month.isin([5, 6, 7, 8, 9, 10])].copy()
    
    synth_data1 = add_residuals(winter[winter.SW_IN_F < 20].copy())
    synth_data2 = add_residuals(winter[winter.SW_IN_F >= 20].copy())
    synth_data3 = add_residuals(summer[summer.SW_IN_F < 20].copy())
    synth_data4 = add_residuals(summer[summer.SW_IN_F >= 20].copy())
    synth_data = pd.concat([synth_data1, synth_data2, synth_data3, synth_data4])

    synth_data = synth_data.sort_index()

    return synth_data
