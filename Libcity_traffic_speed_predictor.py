"""
    Here we take one pair from the displayed pairs of origin and destination id
    After taking input we process it, and we train the traffic speed data of that pair
    of March 2012 las angles. And also test the data and predict the traffic speed between
    that pair for 31st day. And csv file of true value and predicted values is generated.

"""




import sklearn
from numpy import asarray
from xgboost import XGBRegressor
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib
import datetime
import warnings
warnings.filterwarnings('ignore')
from csv import writer
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score


plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

def returnDataFrame(path: str):
    """ Read the csv from given path and return the data """

    data = pd.read_csv(path)
    return data

def pair_data_generator(data, origin, destination):
    """
       It takes origin id(origin) and destination(id) and main dataframe and generate
       csv file for that pair consisting time(31 days), origin_id, destination_id, origin_speed,
       destination_speed
    """
    time = data['time'][:8928:3].values
    origin = [origin] * 2976
    destination = [destination] * 2976
    origin_speed = data.loc[data['entity_id'] == ori]['traffic_speed'][:8928:3].values
    destination_speed = data.loc[data['entity_id'] == dest]['traffic_speed'][:8928:3].values

    dic = {
        'time': time,
        'origin': origin,
        'destination': destination,
        'origin_speed': origin_speed,
        'destination_speed': destination_speed
    }

    # making dataframe
    pair_one_month = pd.DataFrame(
        data=dic,
        index=np.arange(2976)
    )


    # for origin speed
    Q1 = np.percentile(pair_one_month['origin_speed'], 25,
                       interpolation='midpoint')

    Q3 = np.percentile(pair_one_month['origin_speed'], 75,
                       interpolation='midpoint')
    IQR = Q3 - Q1

    # Upper bound
    upper = Q3 + 1.5 * IQR
    upper_array = np.array(pair_one_month['origin_speed'] >= upper)
    # Lower bound
    lower = Q1 - 1.5 * IQR
    lower_array = np.array(pair_one_month['origin_speed'] <= lower)

    pair_one_month.loc[upper_array, 'origin_speed'] = pair_one_month['origin_speed'].mean()
    pair_one_month.loc[lower_array, 'origin_speed'] = pair_one_month['origin_speed'].mean()

    # for destination speed
    Q1 = np.percentile(pair_one_month['destination_speed'], 25,
                       interpolation='midpoint')

    Q3 = np.percentile(pair_one_month['destination_speed'], 75,
                       interpolation='midpoint')
    IQR = Q3 - Q1

    # Upper bound
    upper = Q3 + 1.5 * IQR
    upper_array = np.array(pair_one_month['destination_speed'] >= upper)
    # Lower bound
    lower = Q1 - 1.5 * IQR
    lower_array = np.array(pair_one_month['destination_speed'] <= lower)

    pair_one_month.loc[upper_array, 'destination_speed'] = pair_one_month['destination_speed'].mean()
    pair_one_month.loc[lower_array, 'destination_speed'] = pair_one_month['destination_speed'].mean()
    # # removing outliers
    # pair_one_month.loc[pair_one_month['origin_speed'] < 50, 'origin_speed'] = pair_one_month['origin_speed'].mean()
     #pair_one_month.loc[pair_one_month['destination_speed'] < 50, 'destination_speed'] = pair_one_month['destination_speed'].mean()

    pair_one_month['average_speed'] = (pair_one_month['origin_speed'] + pair_one_month['destination_speed']) / 2

    #saving file
    path = fr'METR_LA/pair_{ori}_{dest}_one_month.csv'
    pair_one_month.to_csv(fr'{path}')
    return path


def getting_required_dataFrame_for_OD(path: str):
    """
       This function takes the origin, destination, and their speed csv file and returns the required dataframe
       that is containing time and average speed of traffic between the origin and destination at that time.
    """
    df = pd.read_csv(path)
    del df['Unnamed: 0']
    time = pd.to_datetime(df['time'])
    required_df = df[['time', 'average_speed']]

    required_df['time'] = pd.to_datetime(required_df['time'])
    required_df = required_df.set_index(required_df['time'])
    required_df = required_df.drop(columns=['time'])
    return required_df


def training_and_testing_df(required_df):
    """

    Args:
        required_df: Dataframe from which training, and testing data is going to be extracted

    Returns:
        train : training data
        test : testing data
    """
    train = required_df['2012-03-01 00:00:00': '2012-03-30 23:45:00'].resample('H').mean()
    test = required_df['2012-03-31 00:00:00' : '2012-03-31 23:45:00'].resample('H').mean()
    return train, test

def order_generator():
    """

    Returns: parameters required for sarimax model fitting

    """
    p = d = q = range(0, 2)

    # take all possible combination for p, d and q
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 24) for x in list(itertools.product(p, d, q))]

    return pdq, seasonal_pdq

def model_fitting(pdq, seasonal_pdq, train):
    """

    Args:
        pdq: p, d, q parameters
        seasonal_pdq: seasonal parameters required for sarimax model
        train: training data

    Returns:
        result: fitting model
    """

    aic_list = []
    req_pdq = 0
    req_spdq = 0
    # Using Grid Search find the optimal set of parameters that yields the best performance
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(train, order=param, seasonal_order=param_seasonal,
                                                enforce_stationary=False, enforce_invertibility=False)
                result = mod.fit()
                aic_list.append(result.aic)
                if min(aic_list) == result.aic:
                    req_pdq = param
                    req_spdq = param_seasonal
                print('ARIMA{}x{}24 - AIC:{}'.format(param, param_seasonal, result.aic))
            except:
                continue

    model = sm.tsa.statespace.SARIMAX(train, order=req_pdq, seasonal_order=req_spdq)
    result = model.fit()
    return result

def graph_plotting(train, prediction, prediction_ci):
    """

    Args:
        train: training data
        prediction: prediction
        prediction_ci: prediction higher lower array

    Returns:
            draw graph of observed plus predicted values
    """


    ax = train['2012-03-01 00:00:00':].plot(label='observed')

    prediction.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=0.4, figsize=(10, 7))
    ax.fill_between(prediction_ci.index, prediction_ci.iloc[:, 0], prediction_ci.iloc[:, 1], color='k', alpha=0.2)
    ax.set_xlabel("Time")
    ax.set_ylabel('Speed')
    plt.legend()
    plt.show()

def error_finding(true_value, predicted_value):
    """

    Args:
        true_value: true testing values
        predicted_value: predicted values

    Returns:
            prints -> Mean absolute percentage error, mean absolute error
    """

    y_pred = predicted_value.predicted_mean[:24]
    y_truth = true_value

    # print(len(y_pred))
    # print(len(y_truth))
    print(f'MAPE : {mean_absolute_percentage_error(y_truth, y_pred)}')
    print(f'MAE : {mean_absolute_error(y_truth, y_pred)}')
    return y_pred, y_truth

def recordWriter(test, y_pred, y_truth, ori, dest,  model_used, csv_path_for_pair):
    y_truth = test
    predicted_Data = pd.DataFrame({
        'time_of_testing': [x for x in test.index],
        'Y_True': [x for x in test.average_speed],
        'Y_Predicted': [x for x in y_pred],

    })

    with open(f'METR_LA/Error_Log.csv', 'a') as f:
        writer_obj = writer(f)
        writer_obj.writerow([ori, dest, mean_absolute_percentage_error(y_truth, y_pred), mean_absolute_error(y_truth, y_pred), model_used])
        f.close()

    if model_used == 'XGBRegressor':
        pred_path = csv_path_for_pair.split('.')[-2][:] + '_XGBRegressor.csv'
    else:
        pred_path = csv_path_for_pair.split('.')[-2][:] + '_Sarimax.csv'

    predicted_Data.to_csv(fr'{pred_path}')

if __name__ == "__main__":
    data = returnDataFrame(r'METR_LA/METR_LA.csv')
    data['time'] = pd.to_datetime(data['time'])
    data = data.iloc[:, 3:]
    # print(data.head())

    data_of_rel = returnDataFrame(r'METR_LA/METR_LA-rel_modified.csv')
    # print(data_of_rel)

    print([[i, j] for i, j in zip(data_of_rel['origin_id'], data_of_rel['destination_id'])])

    ori = int(input('Origin_id : '))
    dest = int(input('Destination_id : '))

    csv_path_for_pair = pair_data_generator(data, ori, dest)

    required_df = getting_required_dataFrame_for_OD(csv_path_for_pair)

    train, test = training_and_testing_df(required_df)


    # print('Using sarimax')
    model_used = 'Sarimax'
    pdq, seasonal_pdq = order_generator()

    result = model_fitting(pdq, seasonal_pdq, train)

    prediction = result.get_prediction(start='2012-03-31 00:00:00', end='2012-03-31 23:00:00', dynamic=False)
    prediction_ci = prediction.conf_int()

    graph_plotting(train, prediction, prediction_ci)

    y_pred, y_truth = error_finding(test, prediction)

    # pred_uc = result.get_forecast(steps=96)
    # pred_ci = pred_uc.conf_int()
    # print(test)
    # print(pred_uc.predicted_mean)

    recordWriter(test, y_pred, y_truth, ori, dest, model_used, csv_path_for_pair)


    model_used = 'XGBRegressor'
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(train, train)
    y_pred = []
    for i in range(len(test)):
        yhat = model.predict(asarray([test.average_speed[i]]))[0]
        y_pred.append(yhat)

    recordWriter(test, y_pred, y_truth, ori, dest, model_used, csv_path_for_pair)

    #
    # # with open('error_records.txt', 'a') as file:
    # #     log = f'\n\nfor pair {ori} - {dest} prediction of 31st of March 2012 based on first 30 days of March\n MAE : {mean_absolute_error(y_truth, y_pred)} \n MAPE : {mean_absolute_percentage_error(y_truth, y_pred)}\n'
    # #     file.write(log)
    #




#
# 'MAPE' : [mean_absolute_percentage_error(y_truth, y_pred)],
#         'MAE' : [mean_absolute_error(y_truth, y_pred)]









