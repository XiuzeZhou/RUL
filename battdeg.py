"""
This module can be used to read cycling data of the CX2, CS2 and PL type cells as
a dataframe. It converts cumulative values into individual values for
each cycle and determines net charge of the battery at every datapoint.
It can also be used to train and test a LSTM model and predict discharge capacity
using the LSTM model.
"""

import datetime
import os
from os import listdir
from os.path import isfile, join
import re
# import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model

# @profile
def date_time_converter(date_time_list):
    """
    This function gets the numpy array with date_time in matlab format
    and returns a numpy array with date_time in human readable format.
    """

    if not isinstance(date_time_list, list):
        raise TypeError("date_time_list should be a list")

    # Empty array to hold the results
    date_time_human = []

    for i in date_time_list:
        date_time_human.append(
            datetime.datetime.fromordinal(
                int(i)) +
            datetime.timedelta(
                days=i %
                1) -
            datetime.timedelta(
                days=366))

    return date_time_human

# @profile


def get_dict_files(data_dir, file_name_format, ignore_file_indices):
    """
    This function finds all the files at the location of the file name
    format as specified and then creates a dictionary after ignoring the
    list of file specified

    Args:
        data_dir (string): This is the absolute path to the data directory.
        file_name_format (string): Format of the filename, used to deduce other
        files.
        ignore_file_indices (list, int): This list of ints tells
        which to ignore.

    Returns:
        The dictionary with all data from files dataframes.
    """

    # get the list of files in the directory
    onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

    # Extract the experiment name from the file_name_format
    exp_name = file_name_format[0:4]

    # Empty dictionary to hold all the dataframe for various files
    dict_files = {}

    # Iterate over all the files of certain type and get the file number from
    # them
    for filename in onlyfiles:
        if exp_name in filename:
            # Extract the filenumber from the name
            file_number = re.search(
                exp_name + r'\((.+?)\).csv',
                filename).group(1)
            # Give a value of dataframe to each key
            dict_files[int(file_number)] = pd.read_csv(
                join(data_dir, filename))

    # Empty dictionary to hold the ordered dictionaries
    dict_ordered = {}
    # Sort the dictionary based on keys
    for key in sorted(dict_files.keys()):
        dict_ordered[key] = dict_files[key]

    # Keys with files to keep, remove the ignore indices from all keys
    wanted_keys = np.array(
        list(set(dict_ordered.keys()) - set(ignore_file_indices)))

    # Remove the ignored dataframes for characterization
    dict_ord_cycling_data = {k: dict_ordered[k] for k in wanted_keys}

    return dict_ord_cycling_data


def concat_dict_dataframes(dict_ord_cycling_data):
    """
    This function takes in a dictionary with ordered keys
    and concatenates the dataframes in the values of the
    dictionary to create a large dataframe with all the records.

    Args:
        dict_ord_cycling_data (dict):
            The dictionary with ordered integer keys and dataframes as values

    Returns:
        The dataframe after concatenation

    """

    # Raise an exception if the type of the inputs is not correct
    if not isinstance(dict_ord_cycling_data, dict):
        raise TypeError('dict_ord_cycling_data is not of type dict')

    #print(dict_ord_cycling_data.keys())
    for i in dict_ord_cycling_data.keys():
        # Raise an exception if the type of the keys is not integers
        # print(type(i))
        if not isinstance(i, (int, np.int64)):
            raise TypeError('a key in the dictionary is not an integer')

    for i in dict_ord_cycling_data.values():
        # Raise an exception if the type of the values is not a dataframe
        if not isinstance(i, pd.DataFrame):
            raise TypeError('a value in the dictionary is not a pandas ' +
                            'dataframe')
        # print(i.columns)
        # Raise am exception if the necessary columns are not found in the df
        if not {
                'Cycle',
                'Charge_Ah',
                'Discharge_Ah',
                'Time_sec',
                'Current_Amp',
                'Voltage_Volt'}.issubset(i.columns):
            raise Exception("the dataframe doesnt have the columns 'Cycle'" +
                            ", 'Charge_Ah', 'Discharge_Ah', " +
                            "'Time_sec', 'Voltage_Volt', 'Current_Amp' ")

    # Concatenate the dataframes to create the total dataframe
    df_out = None
    for k in dict_ord_cycling_data.keys():
        if df_out is None:
            df_next = dict_ord_cycling_data[k]
            df_out = pd.DataFrame(data=None, columns=df_next.columns)
            df_out = pd.concat([df_out, df_next])
        else:
            df_next = dict_ord_cycling_data[k]
            df_next['Cycle'] = np.array(
                df_next['Cycle']) + max(np.array(df_out['Cycle']))
            df_next['Time_sec'] = np.array(
                df_next['Time_sec']) + max(np.array(df_out['Time_sec']))
            df_next['Charge_Ah'] = np.array(
                df_next['Charge_Ah']) + max(np.array(df_out['Charge_Ah']))
            df_next['Discharge_Ah'] = np.array(
                df_next['Discharge_Ah']) + max(np.array(df_out['Discharge_Ah']))
            df_out = pd.concat([df_out, df_next])

    return df_out


def get_cycle_capacities(df_out):
    """
    This function takes the dataframe, creates a new index and then calculates
    capacities per cycle from cumulative charge and discharge capacities

    Args:
        df_out (pandas.DataFrame):
            Concatenated dataframe

    Returns:
        the dataframe with capacities per cycle

    """

    # Raise am exception if the necessary columns are not found in the df
    if not {'Cycle', 'Charge_Ah', 'Discharge_Ah', 'Time_sec', 'Current_Amp',
            'Voltage_Volt'}.issubset(df_out.columns):
        raise Exception("the dataframe doesnt have the columns 'Cycle'" +
                        ", 'Charge_Ah', 'Discharge_Ah', " +
                        "'Time_sec', 'Voltage_Volt', 'Current_Amp' ")

    # Reset the index and drop the old index
    df_out_indexed = df_out.reset_index(drop=True)

    # Proceed further with correcting the capacity
    df_grouped = df_out_indexed.groupby(['Cycle']).count()

    # Get the indices when a cycle starts
    cycle_start_indices = df_grouped['Time_sec'].cumsum()

    # Get the charge_Ah per cycle
    # Create numpy array to store the old charge_Ah row, and then
    # perform transformation on it, rather than in the pandas series
    # this is a lot faster in this case
    charge_cycle_ah = np.array(df_out_indexed['Charge_Ah'])
    charge_ah = np.array(df_out_indexed['Charge_Ah'])

    for i in range(1, len(cycle_start_indices)):
        begin_value = cycle_start_indices.iloc[i - 1]
        end_value = cycle_start_indices.iloc[i]
        charge_cycle_ah[begin_value:end_value] = charge_ah[begin_value:end_value] - \
            charge_ah[begin_value - 1]

    df_out_indexed['charge_cycle_ah'] = charge_cycle_ah

    # Get the discharge_Ah per cycle
    discharge_cycle_ah = np.array(df_out_indexed['Discharge_Ah'])
    discharge_ah = np.array(df_out_indexed['Discharge_Ah'])

    for i in range(1, len(cycle_start_indices)):
        begin_value = cycle_start_indices.iloc[i - 1]
        end_value = cycle_start_indices.iloc[i]
        discharge_cycle_ah[begin_value:end_value] = discharge_ah[begin_value:end_value] - \
            discharge_ah[begin_value - 1]

    df_out_indexed['discharge_cycle_ah'] = discharge_cycle_ah

    # This is the data column we can use for prediction.
    # This is not totally accurate, as this still has some points that go negative,
    # due to incorrect discharge_Ah values every few cycles.
    # But the machine learning algorithm should consider these as outliers and
    # hopefully get over it. We can come back and correct this.
    df_out_indexed['capacity_ah'] = charge_cycle_ah - discharge_cycle_ah
    df_out_indexed.rename(columns={'Current_Amp':'Current(A)','Voltage_Volt':'Voltage(V)'},
                          inplace=True)
    return df_out_indexed

# @profile


def pl_samples_file_reader(data_dir, file_name_format, ignore_file_indices):
    """
    This function reads in the data for PL Samples experiment and returns a
    nice dataframe with cycles in ascending order.

    Args:
        data_dir (string): This is the absolute path to the data directory.
        file_name_format (string): Format of the filename, used to deduce other files.
        ignore_file_indices (list, int): This list of ints tells which to ignore.

    Returns:
        The complete test data in a dataframe with extra column for capacity in Ah.
    """

    # Raise an exception if the type of the inputs is not correct
    if not isinstance(data_dir, str):
        raise TypeError('data_dir is not of type string')

    if not isinstance(file_name_format, str):
        raise TypeError('file_name_format is not of type string')

    if not isinstance(ignore_file_indices, list):
        raise TypeError("ignore_file_indices should be a list")

    for ignore_file_indice in ignore_file_indices:
        if not isinstance(ignore_file_indice, int):
            raise TypeError("""ignore_file_indices elements should be
            of type integer""")

    if not os.path.exists(join(data_dir, file_name_format)):
        raise FileNotFoundError("File {} not found in the location {}"
                                .format(file_name_format, data_dir))

    dict_ord_cycling_data = get_dict_files(
        data_dir, file_name_format, ignore_file_indices)

    df_out = concat_dict_dataframes(dict_ord_cycling_data)

    ####
    # This has been commented out for performance, as we do not need date_time
    ####
    # Convert the Date_Time from matlab datenum to human readable Date_Time
    # First convert the series into a numpy array
    # date_time_matlab = df_out['Date_Time'].tolist()

    # # Apply the conversion to the numpy array
    # df_out['Date_Time_new'] =  date_time_converter(date_time_matlab)

    # Get the cycle capacities from cumulative capacities
    df_out_indexed = get_cycle_capacities(df_out)

    return df_out_indexed

# Wrapping function to train the LSTM model and calculate model_loss,
# and response to the testing data set.


def model_training(data_dir, file_name_format, sheet_name):
    """
    This function converts cumulative battery cycling data into individual cycle data
    and trains the LSTM model with the converted data set.

    Args:
        data_dir (string): This is the absolute path to the data directory.
        file_name_format (string): Format of the filename, used to deduce other files.
        sheet_name(string or int): Sheet name or sheet number in the excel file containing
        the relevant data.

    Returns:
        model_loss(dictionary): Returns the history dictionary (more info to be added)
        y_hat(array): Predicted response for the testing dataset.
        # y_prediction(array): Predicted response for the completely new dataset
        # (The input has to be the time series cycling data including values of
        #  Current, Voltage and Discharge Capacity)
    """
    # The function 'cx2_file_reader' is used to read all the excel files
    # in the given path and convert the given cumulative data into individual
    # cycle data.
    individual_cycle_data = cx2_file_reader(data_dir, file_name_format, sheet_name)

    # The function 'data_formatting' is used to drop the unnecesary columns
    # from the training data i.e. only the features considered in the model
    # (Current, Voltage and Discharge capacity) are retained.
    formatted_data = data_formatting(individual_cycle_data)

    # The function 'series_to_supervised' is used to frame the time series training
    # data as supervised learning dataset.
    learning_df = series_to_supervised(
        formatted_data, n_in=1, n_out=1, dropnan=True)

    # The function 'long_short_term_memory' is used to train the model
    # and predict response for the new input dataset.
    model_loss, y_hat = long_short_term_memory(learning_df)

    return model_loss, y_hat


# Function to predict the discharge capacity using the trained LSTM model.
def model_prediction(input_data):
    """
    This function can be used to forecast the discharge capacity of a battery using
    the trained LSTM model

    Args:
    input_data(dataframe): This is the dataframe containing the current, voltage and
    discharge capacity values at a prior time which can be used to forecast discharge
    capacity at a further time.

    Returns:
    y_predicted: The forecasted values of discharge capacity.
    """

    # The function 'series_to_supervised' is used to frame the time series training
    # data as supervised learning dataset.
    learning_df = series_to_supervised(
        input_data, n_in=1, n_out=1, dropnan=True)
    learning_df = learning_df.iloc[:, 0:3].values
    # Reshaping the input dataset.
    learning_df = learning_df.reshape(
        (learning_df.shape[0], 1, learning_df.shape[1]))
    # Predicting the discharge values using the saved LSTM model.
    module_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = join(module_dir,'models')
    model = load_model(join(model_path,'lstm_trained_model.h5'))
    y_predicted = model.predict(learning_df)
    return y_predicted


# Wrapping function only to merge and convert cumulative data to
# individual cycle data.
def cx2_file_reader(data_dir, file_name_format, sheet_name):
    """
    This function reads in the data for CX2 samples experiment and returns
    a well formatted dataframe with cycles in ascending order.

    Args:
    data_dir (string): This is the absolute path to the data directory.
    file_name_format (string): Format of the filename, used to deduce other files.
    sheet_name (string): Sheet name containing the data in the excel file.

    Returns:
    The complete test data in a dataframe with extra column for capacity in Ah.
    """
    # Raise an exception if the type of the inputs is not correct
    if not isinstance(data_dir, str):
        raise TypeError('data_dir is not of type string')

    if not isinstance(file_name_format, str):
        raise TypeError('file_name_format is not of type string')

    if not isinstance(sheet_name, (str, int)):
        raise TypeError('Sheet_Name format is not of type string or integer')

    if not os.path.exists(join(data_dir, file_name_format)):
        raise FileNotFoundError("File {} not found in the location {}"
                                .format(file_name_format, data_dir))

    # Get the list of files in the directory
    path = join(data_dir, file_name_format)
    files = listdir(path)

    # Extract the experiment name from the file_name_format
    # exp_name = file_name_format[0:6]

    # Filtering out and reading the excel files in the data directory
    file_names = list(filter(lambda x: x[-5:] == '.xlsx', files))

    # Sorting the file names using the
    # 'file_name_sorting' function.
    sorted_name_list = file_name_sorting(file_names)

    # Reading dataframes according to the date of experimentation
    # using 'reading_dataframes' function.
    sorted_df = reading_dataframes(sorted_name_list, sheet_name, path)

    # Merging all the dataframes and adjusting the cycle index
    # using the 'concat_df' function.
    cycle_data = concat_df(sorted_df)

    # Calculating the net capacity of the battery at every datapoint
    # using the function 'capacity'.
    capacity_data = capacity(cycle_data)

    # Returns the dataframe with new cycle indices and capacity data.
    return capacity_data


def file_name_sorting(file_name_list):
    """
    This function sorts all the file names according to the date
    on the file name.

    Args:
    file_name_list(list): List containing all the file names to be read

    Returns:
    A list of file names sorted according to the date on the file name.

    """
    filename = pd.DataFrame(data=file_name_list, columns=['file_name'])
    # Splitting the file name into different columns
    filename['cell_type'], filename['cell_num'], filename['month'], filename[
        'day'], filename['year'] = filename['file_name'].str.split('_', 4).str
    filename['year'], filename['ext'] = filename['year'].str.split('.', 1).str
    filename['date'] = ''
    # Merging the year, month and date column to create a string for DateTime
    # object.
    filename['date'] = filename['year'].map(
        str) + filename['month'].map(str) + filename['day'].map(str)
    # Creating a DateTime object.
    filename['date_time'] = ''
    filename['date_time'] = pd.to_datetime(filename['date'], format="%y%m%d")
    # Sorting the file names according to the
    # created DateTime object.
    filename.sort_values(['date_time'], inplace=True)
    # Created a list of sorted file names
    sorted_file_names = filename['file_name'].values
    return sorted_file_names


def reading_dataframes(file_names, sheet_name, path):
    """
    This function reads all the files in the sorted
    file names list as a dataframe

    Args(list):
    file_names: Sorted file names list
    sheet_name: Sheet name in the excel file containing the data.

    Returns:
    Dictionary of dataframes in the order of the sorted file names.
    """
    # Empty dictionary to store all the dataframes according
    # to the order in the sorted files name list
    df_raw = {}
    # Reading the dataframes
    for i, filename in enumerate(file_names):
        df_raw[i] = pd.read_excel(
            join(
                path,
                filename),
            sheet_name=sheet_name)
    return df_raw


def concat_df(df_dict):
    """
    This function concatenates all the dataframes and edits
    the cycle index for the concatenated dataframes.

    Args:
    df_dict(dictionary): Dictionary of dataframes to be concatenated.

    Returns:
    A concatenated dataframe with editted cycle index

    """
    df_concat = None
    for data in df_dict:
        if df_concat is None:
            df_next = df_dict[data]
            df_concat = pd.DataFrame(data=None, columns=df_next.columns)
            # df_next['Cycle'] = df_next['Cycle'] + max(df_pl12['Cycle'])
            df_concat = pd.concat([df_concat, df_next])
        else:
            df_next = df_dict[data]
            df_next['Cycle_Index'] = np.array(
                df_next['Cycle_Index']) + max(np.array(df_concat['Cycle_Index']))
            df_next['Test_Time(s)'] = np.array(
                df_next['Test_Time(s)']) + max(np.array(df_concat['Test_Time(s)']))
            df_next['Charge_Capacity(Ah)'] = np.array(
                df_next['Charge_Capacity(Ah)']) + max(np.array(df_concat['Charge_Capacity(Ah)']))
            df_next['Discharge_Capacity(Ah)'] = np.array(
                df_next['Discharge_Capacity(Ah)']) + max(
                    np.array(df_concat['Discharge_Capacity(Ah)']))
            df_concat = pd.concat([df_concat, df_next])
    # Reset the index and drop the old index
    df_reset = df_concat.reset_index(drop=True)
    return df_reset


def capacity(df_data):
    """
    This function calculates the net capacity of the battery
    from the charge capacity and discharge capacity values.

    Args:
    df_data(dataframe): Concatenated dataframe which has the values of charge
    capacity and discharge capacity for which net capacity has to be
    calculated.

    Returns:
    Dataframe with net capacity of the battery for every point of the charge
    and discharge cycle.
    """
    # Grouping rows by the cycle index.
    group = df_data.groupby(['Cycle_Index']).count()

    # Get the indices when a cycle starts
    cycle_start_indices = group['Data_Point'].cumsum()

    # Get the charge_Ah per cycle
    # Create numpy array to store the old charge_Ah row, and then
    # perform transformation on it, rather than in the pandas series
    # this is a lot faster in this case
    charge_cycle_ah = np.array(df_data['Charge_Capacity(Ah)'])
    charge_ah = np.array(df_data['Charge_Capacity(Ah)'])

    for i in range(1, len(cycle_start_indices)):
        begin_value = cycle_start_indices.iloc[i - 1]
        end_value = cycle_start_indices.iloc[i]
        charge_cycle_ah[begin_value:end_value] = charge_ah[begin_value:end_value] - \
            charge_ah[begin_value - 1]

    df_data['charge_cycle_ah'] = charge_cycle_ah

    # Get the discharge_Ah per cycle
    discharge_cycle_ah = np.array(df_data['Discharge_Capacity(Ah)'])
    discharge_ah = np.array(df_data['Discharge_Capacity(Ah)'])

    for i in range(1, len(cycle_start_indices)):
        begin_value = cycle_start_indices.iloc[i - 1]
        end_value = cycle_start_indices.iloc[i]
        discharge_cycle_ah[begin_value:end_value] = discharge_ah[begin_value:end_value] - \
            discharge_ah[begin_value - 1]

    df_data['discharge_cycle_ah'] = discharge_cycle_ah

    # This is the data column we can use for prediction.
    # This is not totally accurate, as this still has some points that go negative,
    # due to incorrect discharge_Ah values every few cycles.
    # But the machine learning algorithm should consider these as outliers and
    # hopefully get over it. We can come back and correct this.
    df_data['capacity_ah'] = df_data['charge_cycle_ah'] - df_data['discharge_cycle_ah']

    return df_data


def data_formatting(merged_df):
    """
    This function formats the merged dataframe so that it can be used to frame the given
    time series data as a supervised learning dataset.

    Args:
        merged_df(dataframe): The merged dataframe which can be obtained by using the
        function 'cx2_file_reader'

    Returns:
        A numpy array with only values required to frame a time series as a
        supervised learning dataset.
    """
    # Get the columns containing text 'Current', 'Voltage' and
    # 'discharge_cycle_ah'
    merged_df = merged_df.filter(regex='Current|Voltage|discharge_cycle_ah')
    formatted_df = merged_df.astype('float32')
    return formatted_df


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    
    """
    n_vars = 1 if isinstance(data, list) else data.shape[1]
    df_data = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df_data.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df_data.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    sl_df = pd.concat(cols, axis=1)
    sl_df.columns = names
    # drop rows with NaN values
    if dropnan:
        sl_df.dropna(inplace=True)
    sl_df.drop(sl_df.columns[[3, 4]], axis=1, inplace=True)
    sl_df.rename(columns={'var1(t-1)':'Current(t-1)','var2(t-1)':'Voltage(t-1)',
                 'var3(t-1)':'discharge_capacity(t-1)','var3(t)':'discharge_capacity(t)'},
                 inplace = True)
    return sl_df


def long_short_term_memory(model_data):
    """
    This function splits the input dataset into training
    and testing datasets. The keras LSTM model is then
    trained and tested using the respective datasets.

    Args:
        model_data(dataframe): Values of input and output variables
        of time series data framed as a supervised learning dataset.


    Returns:
        model_loss(dictionary): Returns the history dictionary (more info to be added)
        y_hat(array): Predicted response for the testing dataset.
        y_prediction(array): Predicted response for the completely new dataset.
    """
    # Splitting the input dataset into training and testing data
    train, test = train_test_split(model_data, test_size=0.2, random_state=944)
    # split into input and outputs
    train_x, train_y = train[train.columns[0:3]
                             ].values, train[train.columns[3]].values
    test_x, test_y = test[test.columns[0:3]
                          ].values, test[test.columns[3]].values
    # reshape input to be 3D [samples, timesteps, features]
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    # Designing the network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # Fitting the network with training and testing data
    history = model.fit(
        train_x,
        train_y,
        epochs=50,
        batch_size=72,
        validation_data=(
            test_x,
            test_y),
        verbose=0,
        shuffle=False)
    model_loss = history.history
    # Prediction for the test dataset.
    yhat = model.predict(test_x)
    # model.save('lstm_trained_model.h5')
    return model_loss, yhat

def file_reader(data_dir, file_name_format, sheet_name, ignore_file_indices):
    """
    This function reads PL sample, CX2 and CS2 files and returns a nice 
    dataframe with cyclic values of charge and discharge capacity with 
    cycles in ascending order
    
    Args:
    data_dir (string): This is the absolute path to the data directory.
    file_name_format (string): Format of the filename, used to deduce other files.
    sheet_name (string): Sheet name containing the data in the excel file.
    ignore_file_indices (list, int): This list of ints tells which to ignore.

    Returns:
    The complete test data in a dataframe with extra column for capacity in Ah.
    """

    # For excel files (CX2 and CS2 datafiles), the function 'cx2_file_reader'
    # is used.
    if file_name_format[:3] == 'CX2' or file_name_format[:3] == 'CS2':
        df_output = cx2_file_reader(data_dir,file_name_format,sheet_name) 
    else:
        df_output = pl_samples_file_reader(data_dir,file_name_format,ignore_file_indices)
   
    # The function 'data_formatting' is used to drop the unnecesary columns
    # from the training data i.e. only the features considered in the model
    # (Current, Voltage and Discharge capacity) are retained.
    formatted_data = data_formatting(df_output)

    # The function 'series_to_supervised' is used to frame the time series training
    # data as supervised learning dataset.
    # df_out = series_to_supervised(
    #     formatted_data, n_in=1, n_out=1, dropnan=True)
    return formatted_data
