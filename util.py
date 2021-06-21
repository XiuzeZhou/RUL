import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler

def getBatteryCapacity(Battery):
    cycle = []
    capacity = []
    i = 1
    for Bat in Battery:
        if Bat['cycle'] == 'discharge':
            cycle.append(i)
            capacity.append(Bat['data']['Capacity'][0])
            i += 1
    return [cycle, capacity]

def getChargingValues(Battery, Index):
    Battery = Battery[Index]['data']
    index = []
    i = 1
    for iterator in Battery['Voltage_measured']:
        index.append(i)
        i += 1
    return [index, Battery['Voltage_measured'], Battery['Current_measured'], Battery['Temperature_measured'], Battery['Voltage_charge'], Battery['Time']]

def getDischargingValues(Battery, Index):
    Battery = Battery[Index]['data']
    index = []
    i = 1
    for iterator in Battery['Voltage_measured']:
        index.append(i)
        i += 1
    return [index, Battery['Voltage_measured'], Battery['Current_measured'], Battery['Temperature_measured'], Battery['Voltage_load'], Battery['Time']]

def getMaxDischargeTemp(Battery):
    cycle = []
    temp = []
    i = 1
    for Bat in Battery:
        if Bat['cycle'] == 'discharge':
            cycle.append(i)
            temp.append(max(Bat['data']['Temperature_measured']))
            i += 1
    return [cycle, temp]

def getMaxChargeTemp(Battery, discharge_len):
    cycle = []
    temp = []
    i = 1
    for Bat in Battery:
        if Bat['cycle'] == 'charge':
            cycle.append(i)
            temp.append(max(Bat['data']['Temperature_measured']))
            i += 1
    return [cycle[:discharge_len], temp[:discharge_len]]

def getDataframe(Battery):
    l = getBatteryCapacity(Battery)
    l1 = getMaxDischargeTemp(Battery)
    l2 = getMaxChargeTemp(Battery, len(l1[0]))
    data = {'cycle':l[0],'capacity':l[1], 'max_discharge_temp':l1[1], 'max_charge_temp':l2[1]}
    return pd.DataFrame(data)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def supervisedDataframeBuilder(Batterydataframe, scaler):
    values = Batterydataframe[['capacity']]
    scaled = scaler.fit_transform(values)
    data = series_to_supervised(scaled, 5, 1)
    data['cycle'] = data.index

    return data

def splitDataFrame(Dataframe, ratio):
    X = Dataframe[['cycle', 'var1(t-5)', 'var1(t-4)', 'var1(t-3)', 'var1(t-2)', 'var1(t-1)']]
    Y = Dataframe[['var1(t)']]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = ratio, shuffle=False)
    
    return X_train, X_test, y_train, y_test

def moving_average(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')

def rollingAverage(x_stuff, y_stuff):
    window_size = 10
    sigma=1.0
    avg = moving_average(y_stuff, window_size)
    avg_list = avg.tolist()
    residual = y_stuff - avg
    testing_std = residual.rolling(window_size).std()
    testing_std_as_df = pd.DataFrame(testing_std)
    rolling_std = testing_std_as_df.replace(np.nan,
                  testing_std_as_df.iloc[window_size - 1]).round(3).iloc[:,0].tolist()
    rolling_std
    std = np.std(residual)
    lst=[]
    lst_index = 0
    lst_count = 0
    for i in y_stuff.index:
        if (y_stuff[i] > avg_list[lst_index] + (1.5 * rolling_std[lst_index])) | (y_stuff[i] < avg_list[lst_index] - (1.5 * rolling_std[lst_index])):
            lt=[i,x_stuff[i], y_stuff[i],avg_list[lst_index],rolling_std[lst_index]]
            lst.append(lt)
            lst_count+=1
        lst_index+=1

    lst_x = []
    lst_y = []

    for i in range (0,len(lst)):
        lst_x.append(lst[i][1])
        lst_y.append(lst[i][2])

    return lst_x, lst_y

