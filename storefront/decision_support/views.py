import tensorflow 
from keras.models import load_model
from django.shortcuts import render
#from django.http import HttpResponse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#from keras.models import Sequential
#from keras.layers import LSTM, Dense
from keras.layers import Input
import mpld3
from mpld3 import plugins
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import load_model
import datetime
import matplotlib.dates as mdates
import csv



model_water = load_model('Model_water.h5')

first_data = pd.read_csv('water_data.csv')
first_data['Rainfall'] = pd.to_numeric(first_data['Rainfall'], errors='coerce')

# Split data into training and testing sets
train_size = int(len(first_data) * 0.8)
train_data = first_data.iloc[:train_size]

# Calculate means from training data
water_mean = train_data['Water Level'].mean()
drawdown_mean = train_data['Drawdown'].mean()

# Fill missing values with means
data = first_data.fillna(value={'Water Level': water_mean, 'Rainfall': 0, 'Drawdown': drawdown_mean}).copy()

# Convert 'Date' column to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
data.set_index('Date', inplace=True)

# Scale the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :])
        y.append(data[i+seq_length, 0])
    return np.array(X), np.array(y)

seq_length = 30
X_train, y_train = create_sequences(scaled_data[:train_size], seq_length)
X_test, y_test = create_sequences(scaled_data[train_size:], seq_length)



train_dates = list(data.index)
n_past = 15
n_days_for_prediction= 480
predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq='d').tolist()   
prediction = model_water.predict(X_test[-n_days_for_prediction:]) 
prediction_copies = np.repeat(prediction, data.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:, 0]
forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())

df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Water Level':y_pred_future})

df_forecast.set_index('Date', inplace=True)


# past data
original = data[['Water Level']]
original = original.loc[(original.index >= original.index[-7]) & (original.index <= original.index[-1])]

# forecasted data plot
last_known_date = original.index[-1]
start_date = last_known_date + pd.Timedelta(days=-6)
forecast_end_date = start_date + pd.Timedelta(days=30)
forecast_dates = pd.date_range(start=start_date, end=forecast_end_date)
forecast_values = df_forecast.loc[forecast_dates, 'Water Level']

# Connecting the plot from the last data in csv to the forecasted water level
last_known_value = original['Water Level'].iloc[-1]

now = datetime.datetime.now()
dateToday = now.strftime("%A %d %B, %Y  %I:%M%p")


def Dashboard(request):
    forecasted_tom =  forecast_values.iloc[0]
    Yesterday = original['Water Level'].iloc[-2]
    specific_date_last_year = "2022-Dec-31"
    last_year_date = datetime.datetime.strptime(specific_date_last_year, "%Y-%b-%d").date()
    last_year_timestamp = pd.Timestamp(last_year_date)
    last_year_value = data.loc[last_year_timestamp, 'Water Level']
    date_last_year = last_year_timestamp.strftime("%B %d, %Y")
    min_water_level = data['Water Level'].min()


    def water_level_dashboard():
        # past data
        original = data[['Water Level']]
        original = original.loc[(original.index >= original.index[-7]) & (original.index <= original.index[-1])]

        # forecasted data plot
        last_known_date = original.index[-1]
        start_date = last_known_date + pd.Timedelta(days=-6)
        forecast_end_date = start_date + pd.Timedelta(days=7)
        forecast_dates = pd.date_range(start=start_date, end=forecast_end_date)
        forecast_values = df_forecast.loc[forecast_dates, 'Water Level']
        # Plotting
        fig, ax = plt.subplots(figsize=(5.5, 4.4))

        # Plot past water level
        past_plot = ax.plot(original.index, original['Water Level'], color='#7CFC00', marker='o', markersize=7, label='Past Water Level', linewidth=3)

        
        forecast_plot = ax.plot(forecast_values.index, forecast_values, label='Forecasted Water Level', color='orange', marker='o', markersize=7, linewidth=3)
    
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        
        ax.set_xlabel('DATE', fontsize=14, color="white", labelpad=5)
        ax.set_ylabel('WATER LEVEL', fontsize=15, color="white", labelpad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.yaxis.set_tick_params(labelcolor='black', labelsize=10)
        ax.xaxis.set_tick_params(rotation=85, labelcolor='black', labelsize=10)
        ax.grid(axis='both', linestyle='--', alpha=0.2)
        ax.legend(loc='lower right', fontsize='medium', ncol=1, borderpad=0.5, borderaxespad=0.5, shadow=True)
        tooltip1 = plugins.PointHTMLTooltip(past_plot[0], labels=list(original['Water Level'].values), voffset=20, hoffset=10)
        tooltip2 = plugins.PointHTMLTooltip(forecast_plot[0], labels=list(forecast_values.values), voffset=20, hoffset=10)

        plugins.connect(fig, tooltip1)
        plugins.connect(fig, tooltip2)


        plt.tight_layout()

        # Convert plot to HTML using mpld3
        html_str = mpld3.fig_to_html(fig)

        plt.close()  # Close the plot to free up memory

        return html_str

        # plt.savefig('decision_support/static/img/water_level_dashboard.png', transparent=True)
    plot = water_level_dashboard()


    def minimum_water_level(csv_file):
        min_water_level = float('inf')
        min_water_level_date = None

        with open(csv_file, 'r', newline='') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                water_level_str = row['Water Level']
                if water_level_str:  
                    try:
                        water_level = float(water_level_str)
                        if water_level < min_water_level:
                            min_water_level = water_level
                            min_water_level_date = row['Date']
                    except ValueError:
                        pass
            return min_water_level, min_water_level_date

    csv_file = 'water_data.csv' 

    min_water_level, min_water_level_date = minimum_water_level(csv_file)
    min_year_date = datetime.datetime.strptime(min_water_level_date, "%d-%b-%y").date()
    min_year_timestamp = pd.Timestamp(min_year_date)
    min_year_value = data.loc[min_year_timestamp, 'Water Level']
    min_water_level_date = min_year_timestamp.strftime("%B %d, %Y")

    def water_alloc():
        plt.cla()
        data = pd.read_csv('manila_water_data.csv')
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
        data = data.tail(6)
        y = data['Business Zone']
        x = data['Supply Volume']
        plt.figure(figsize=(15, 6))
        bars = plt.barh(y, x, color='skyblue', edgecolor='black', linewidth=0.5)

        for bar, val in zip(bars, x):
            plt.text(bar.get_width() - 0.2, bar.get_y() + bar.get_height()/2, str(val), color='black', va='center', ha='right')

        plt.xlabel('Supply Volume', color='white')
        plt.ylabel('Business Zones', color='white')

        plt.xticks(size='large', color='white')
        plt.yticks(size='large', color='white')
        plt.savefig('decision_support/static/img/water_alloc.png', transparent=True)
    water_alloc()

    date_today = df_forecast.index[13]
    date_yest = df_forecast.index[14]
    date_tom = df_forecast.index[15]
    return render(request, 'Dashboard.html', 
                  {'Tomorrow': forecasted_tom, 
                   'Today': last_known_value, 
                   'Yesterday': Yesterday,
                   'last_year_today': last_year_value,
                   'date_last_year': date_last_year,
                   'min_water_level': min_water_level,
                   'min_water_level_date': min_water_level_date,
                   'Date': dateToday,
                   'date_today': date_today,
                   'date_yest': date_yest,
                   'date_tom': date_tom,
                   'plot': plot})




def Forecast(request):
    def water_level_plot():
        plt.cla()
        title = 'Forecasted Water Level'
        plt.figure(figsize=(20, 10))
        plt.plot(original.index, original['Water Level'], color='#7CFC00', marker='o', markersize=8, label='Past Water Level', linewidth=3)
        plt.plot(forecast_values.index, forecast_values, label='Forecasted Water Level', color='orange', marker='o', markersize=8, linewidth=3)

        # Connect past data to forecasted data
        #plt.plot([last_known_date, forecast_dates[0]], [last_known_value, forecast_values.iloc[0]], color='green', linestyle='--')
        plt.xlabel('DATE', fontsize=20, color="white", labelpad=30)  
        plt.ylabel('WATER LEVEL', fontsize=20, color="white", labelpad=30) 
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.yticks(color='white',fontsize=20)
        plt.xticks(rotation=85, color='white', fontsize=20)
        plt.grid(axis='x', linestyle='--', alpha=0.2)
        plt.grid(axis='y', linestyle='--', alpha=0.2)
        plt.legend(loc='upper right', fontsize='xx-large', ncol=1, borderpad=1.0, borderaxespad=1.0, shadow=True)      
        plt.gca().spines['top'].set_color('black')  
        plt.gca().spines['right'].set_color('black')  
        plt.gca().spines['bottom'].set_color('black')  
        plt.gca().spines['left'].set_color('black')  
        plt.tight_layout() 
        plt.savefig('decision_support/static/img/water_level_plot.png', transparent=True)
    forecasted_date = df_forecast.index[14]
    forecasted = df_forecast['Water Level'].iloc[14]
    forecasted = round(forecasted, 2)

    # water_level_plot()
    # PREDICTION
    train_predictions = model_water.predict(X_train)
    test_predictions = model_water.predict(X_test)
    # SCALING FOR PREDICTION
    train_predictions = scaler.inverse_transform(np.concatenate((train_predictions, X_train[:, -1, 1:]), axis=1))[:, 0]
    test_predictions = scaler.inverse_transform(np.concatenate((test_predictions, X_test[:, -1, 1:]), axis=1))[:, 0]

    # SCALING FOR ACTUAL
    y_train_inv = scaler.inverse_transform(np.concatenate((y_train.reshape(-1, 1), X_train[:, -1, 1:]), axis=1))[:, 0]
    y_test_inv = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), X_test[:, -1, 1:]), axis=1))[:, 0]

    # sMAPE of actual data to forecasted data
    test_predictions = np.array(test_predictions)
    y_test_inv = np.array(y_test_inv)
    fore_error = abs(y_test_inv - test_predictions)
    fore_percentage_error = fore_error / (abs(y_test_inv) + abs(test_predictions))
    fore_smape = 100 * np.mean(fore_percentage_error)
    fore_smape = round(fore_smape,2)
    #fore_smape = np.mean((np.abs(test_predictions - y_test_inv) / np.abs(test_predictions + y_test_inv))) * 100
    #fore_smape = 100 - fore_smape

    # sMAPE of actual data to forecasted data
    actual_error = abs(original['Water Level'].iloc[-1] - df_forecast['Water Level'].iloc[14])
    act_percentage_error = actual_error / (abs(original['Water Level'].iloc[-1]) + abs(df_forecast['Water Level'].iloc[14]))
    act_smape = 100 * np.mean(act_percentage_error)
    act_smape = round(act_smape,2)


    #act_smape = np.mean((np.abs(df_forecast['Water Level'].iloc[14] - original['Water Level'].iloc[-1]) / np.abs(df_forecast['Water Level'].iloc[14] + original['Water Level'].iloc[-1])/2)) * 100
    #act_smape = round(act_smape,2)
    #act_smape = 100 - act_smape




    def rainfall_plot(fore_rain_smape, act_rain_smape, forecast_rain, actual_rain):
        model_rainfall = load_model('Model_rainfall.h5')
        first_data = pd.read_csv('rainfall_data.csv')
        first_data['RAINFALL'] = pd.to_numeric(first_data['RAINFALL'], errors='coerce')

        # Split data into training and testing sets
        train_size = int(len(first_data) * 0.8)
        train_data = first_data.iloc[:train_size]
        test_data = first_data.iloc[train_size:]

        tmax_mean = train_data['TMAX'].mean()
        tmin_mean = train_data['TMIN'].mean()
        tmean_mean = train_data['TMEAN'].mean()
        wind_speed_mean = train_data['WIND_SPEED'].mean()
        wind_direct_mean = train_data['WIND_DIRECTION'].mean()
        rh_mean = train_data['RH'].mean()

        data = first_data.fillna(value={'RAINFALL': 0, 'TMAX': tmax_mean, 'TMIN': tmin_mean, 'TMEAN': tmean_mean, 'WIND_SPEED': wind_speed_mean, 'WIND_DIRECTION': wind_direct_mean, 'RH': rh_mean}).copy()    
        data['Date'] = pd.to_datetime(data[['YEAR', 'MONTH', 'DAY']], format='%d-%b-%y')
        data.set_index('Date', inplace=True)
        data.drop(columns=['YEAR', 'DAY', 'MONTH'], inplace=True)

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        def create_sequences(data, seq_length):
            X = []
            y = []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length, :])
                y.append(data[i+seq_length, 0])
            return np.array(X), np.array(y)

        seq_length = 100
        X_train, y_train = create_sequences(scaled_data[:train_size], seq_length)
        X_test, y_test = create_sequences(scaled_data[train_size:], seq_length)

        train_dates = list(data.index)
        n_past = 10
        n_days_for_prediction= 380
        predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq='d').tolist()
        prediction = model_rainfall.predict(X_train[-n_days_for_prediction:]) 
        prediction_copies = np.repeat(prediction, data.shape[1], axis=-1)
        y_pred_future = scaler.inverse_transform(prediction_copies)[:, 0]
        forecast_dates = []
        for time_i in predict_period_dates:
            forecast_dates.append(time_i.date())
            
        df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'RAINFALL':y_pred_future})
        df_forecast.set_index('Date', inplace=True)

        # For the past data
        original = data[['RAINFALL']]
        original = original.loc[(original.index >= original.index[-7]) & (original.index <= original.index[-1])]

        # For the forecasted data plot
        last_known_date = original.index[-1]
        start_date = last_known_date + pd.Timedelta(days=-6)
        forecast_end_date = start_date + pd.Timedelta(days=30)
        forecast_dates = pd.date_range(start=start_date, end=forecast_end_date)
        forecast_values = df_forecast.loc[forecast_dates, 'RAINFALL']

        # Connect the past data to the forecasted data
        last_known_value = original['RAINFALL'].iloc[-1]

        plt.cla()
        title = 'Forecasted Rainfall'
        plt.figure(figsize=(20, 10))
        plt.plot(original.index, original['RAINFALL'], color='#7CFC00', marker='o', markersize=8, label='Past Rainfall', linewidth=3)
        plt.plot(forecast_values.index, forecast_values, label='Forecasted Rainfall', color='orange', marker='o', markersize=8, linewidth=3)

        # Connect past data to forecasted data
        #plt.plot([last_known_date, forecast_dates[0]], [last_known_value, forecast_values.iloc[0]], color='green', linestyle='--')
        plt.xlabel('DATE', fontsize=20, color="white", labelpad=30)  
        plt.ylabel('RAINFALL', fontsize=20, color="white", labelpad=30) 
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.yticks(color='white',fontsize=20)
        plt.xticks(rotation=85, color='white', fontsize=20)
        plt.grid(axis='x', linestyle='--', alpha=0.2)
        plt.grid(axis='y', linestyle='--', alpha=0.2)
        plt.legend(loc='upper right', fontsize='xx-large', ncol=1, borderpad=1.0, borderaxespad=1.0, shadow=True)      
        plt.gca().spines['top'].set_color('black')  
        plt.gca().spines['right'].set_color('black')  
        plt.gca().spines['bottom'].set_color('black')  
        plt.gca().spines['left'].set_color('black')  
        plt.tight_layout() 
        plt.savefig('decision_support/static/img/rainfall_plot.png', transparent=True)

        # PREDICTION
        train_predictions = model_rainfall.predict(X_train)
        test_predictions = model_rainfall.predict(X_test)
        # SCALING FOR PREDICTION
        train_predictions = scaler.inverse_transform(np.concatenate((train_predictions, X_train[:, -1, 1:]), axis=1))[:, 0]
        test_predictions = scaler.inverse_transform(np.concatenate((test_predictions, X_test[:, -1, 1:]), axis=1))[:, 0]

        # SCALING FOR ACTUAL
        y_train_inv = scaler.inverse_transform(np.concatenate((y_train.reshape(-1, 1), X_train[:, -1, 1:]), axis=1))[:, 0]
        y_test_inv = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), X_test[:, -1, 1:]), axis=1))[:, 0]

        # sMAPE of actual data to forecasted data
        test_predictions = np.array(test_predictions)
        y_test_inv = np.array(y_test_inv)

        fore_rain_error = abs(y_test_inv - test_predictions)
        fore_percentage_rain_error = fore_rain_error / (abs(y_test_inv) + abs(test_predictions))
        fore_rain_smape = 100 * np.mean(fore_percentage_rain_error)
        fore_rain_smape = round(fore_rain_smape,2)
        #fore_smape = np.mean((np.abs(test_predictions - y_test_inv) / np.abs(test_predictions + y_test_inv))) * 100
        #fore_rain_smape = 100 - fore_rain_smape


        # sMAPE of actual data to forecasted data
        actual_rain_error = abs(original['RAINFALL'].iloc[-1] - df_forecast['RAINFALL'].iloc[9])
        act_percentage_rain_error = actual_rain_error / (abs(original['RAINFALL'].iloc[-1]) + abs(df_forecast['RAINFALL'].iloc[9]))
        act_rain_smape = 100 * np.mean(act_percentage_rain_error)
        #act_rain_smape = 100 - act_rain_smape
        act_rain_smape = round(act_rain_smape,2)

        #fore_rain_smape = np.mean((np.abs(test_predictions - y_test_inv) / np.abs(test_predictions + y_test_inv))) * 100
        #fore_rain_smape = round(fore_rain_smape,2)
        #fore_rain_smape = 100 - fore_rain_smape
        forecast_rain = df_forecast['RAINFALL'].iloc[9]
        forecast_rain = round(forecast_rain,2)
        #fore_rain_smape = 100 - fore_rain_smape

        # sMAPE of actual data to forecasted data
        #act_rain_smape = np.mean((np.abs(df_forecast['RAINFALL'].iloc[9] - original['RAINFALL'].iloc[-1]) / np.abs(df_forecast['RAINFALL'].iloc[9] + original['RAINFALL'].iloc[-1]))) * 100
        #act_rain_smape = round(act_rain_smape,2)
        #act_rain_smape = 100 - act_rain_smape
        actual_rain = original['RAINFALL'].iloc[-1]
        

        return fore_rain_smape, act_rain_smape, forecast_rain, actual_rain

    forecast_rain = 0
    actual_rain = 0
    fore_rain_smape = 0.00
    act_rain_smape = 0.00
    # fore_rain_smape, act_rain_smape, forecast_rain, actual_rain = rainfall_plot(fore_rain_smape, act_rain_smape, forecast_rain, actual_rain)

    def drawdown_plot(fore_drawdown_smape, act_drawdown_smape, forecast_drawdown, actual_drawdown):
        model_drawdown = load_model('Model_drawdown.h5')
        first_data = pd.read_csv('water_data.csv')
        columns = ['Date', 'Drawdown', 'Rainfall', 'Water Level'] 
        first_data = first_data[columns]
        first_data['Rainfall'] = pd.to_numeric(first_data['Rainfall'], errors='coerce')

        # Split data into training and testing sets
        train_size = int(len(first_data) * 0.8)
        train_data = first_data.iloc[:train_size]
        test_data = first_data.iloc[train_size:]

        # Calculate means from training data
        water_mean = train_data['Water Level'].mean()
        drawdown_mean = train_data['Drawdown'].mean()

        # Fill missing values with means
        data = first_data.fillna(value={'Drawdown': drawdown_mean, 'Rainfall': 0, 'Water Level': water_mean,}).copy()

        # Convert 'Date' column to datetime and set as index
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
        data.set_index('Date', inplace=True)

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        def create_sequences(data, seq_length):
            X = []
            y = []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length, :])
                y.append(data[i+seq_length, 0])
            return np.array(X), np.array(y)

        seq_length = 50
        X_train, y_train = create_sequences(scaled_data[:train_size], seq_length)
        X_test, y_test = create_sequences(scaled_data[train_size:], seq_length)



        train_dates = list(data.index)
        n_past = 7
        n_days_for_prediction= 350
        
        predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq='d').tolist()
        prediction = model_drawdown.predict(X_test[-n_days_for_prediction:]) 
        prediction_copies = np.repeat(prediction, data.shape[1], axis=-1)
        y_pred_future = scaler.inverse_transform(prediction_copies)[:, 0]
        forecast_dates = []
        for time_i in predict_period_dates:
            forecast_dates.append(time_i.date())
            
        df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Drawdown':y_pred_future})
        df_forecast.set_index('Date', inplace=True)
        original = data[['Drawdown']]
        actual_drawdown = original['Drawdown'].iloc[-1]
        original = original.loc[(original.index >= original.index[-7]) & (original.index <= original.index[-1])]

        # For the forecasted data plot
        last_known_date = original.index[-1]
        start_date = last_known_date + pd.Timedelta(days=-6)
        forecast_end_date = start_date + pd.Timedelta(days=30)
        forecast_dates = pd.date_range(start=start_date, end=forecast_end_date)
        forecast_values = df_forecast.loc[forecast_dates, 'Drawdown']

        # Connect the past data to the forecasted data
        last_known_value = original['Drawdown'].iloc[-1]

        plt.cla()
        title = 'Forecasted Drawdown'
        plt.figure(figsize=(20, 10))
        plt.plot(original.index, original['Drawdown'], color='#7CFC00', marker='o', markersize=8, label='Past Drawdown', linewidth=3)
        plt.plot(forecast_values.index, forecast_values, label='Forecasted Drawdown', color='orange', marker='o', markersize=8, linewidth=3)

        # Connect past data to forecasted data
        # plt.plot([last_known_date, forecast_dates[0]], [last_known_value, forecast_values.iloc[0]], color='green', linestyle='--')
        plt.xlabel('DATE', fontsize=20, color="white", labelpad=30)  
        plt.ylabel('Drawdown', fontsize=20, color="white", labelpad=30) 
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.yticks(color='white',fontsize=20)
        plt.xticks(rotation=85, color='white', fontsize=20)
        plt.grid(axis='x', linestyle='--', alpha=0.2)
        plt.grid(axis='y', linestyle='--', alpha=0.2)
        plt.legend(loc='upper right', fontsize='xx-large', ncol=1, borderpad=1.0, borderaxespad=1.0, shadow=True)      
        plt.gca().spines['top'].set_color('black')  
        plt.gca().spines['right'].set_color('black')  
        plt.gca().spines['bottom'].set_color('black')  
        plt.gca().spines['left'].set_color('black')  
        plt.tight_layout() 
        plt.savefig('decision_support/static/img/drawdown_plot.png', transparent=True)

         # PREDICTION
        train_predictions = model_drawdown.predict(X_train)
        test_predictions = model_drawdown.predict(X_test)
        # SCALING FOR PREDICTION
        train_predictions = scaler.inverse_transform(np.concatenate((train_predictions, X_train[:, -1, 1:]), axis=1))[:, 0]
        test_predictions = scaler.inverse_transform(np.concatenate((test_predictions, X_test[:, -1, 1:]), axis=1))[:, 0]

        # SCALING FOR ACTUAL
        y_train_inv = scaler.inverse_transform(np.concatenate((y_train.reshape(-1, 1), X_train[:, -1, 1:]), axis=1))[:, 0]
        y_test_inv = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), X_test[:, -1, 1:]), axis=1))[:, 0]

        # sMAPE of actual data to forecasted data
        test_predictions = np.array(test_predictions)
        y_test_inv = np.array(y_test_inv)
        fore_drawdown_smape = np.mean((np.abs(test_predictions - y_test_inv) / np.abs(test_predictions + y_test_inv)/2)) * 100
        #fore_drawdown_smape = 200 - fore_drawdown_smape
        fore_drawdown_smape = round(fore_drawdown_smape,2)

        # sMAPE of actual data to forecasted data
        act_drawdown_smape = np.mean((np.abs(df_forecast['Drawdown'].iloc[4] - original['Drawdown'].iloc[-1]) / np.abs(df_forecast['Drawdown'].iloc[4] + original['Drawdown'].iloc[-1])/2)) * 100
        #act_drawdown_smape = 200 - act_drawdown_smape
        act_drawdown_smape = round(act_drawdown_smape,2)
        forecast_drawdown = df_forecast['Drawdown'].iloc[4]
        return fore_drawdown_smape, act_drawdown_smape, forecast_drawdown, actual_drawdown
    
    forecast_drawdown = 0
    actual_drawdown = 0
    fore_drawdown_smape = 0.00
    act_drawdown_smape = 0.00
    # fore_drawdown_smape, act_drawdown_smape, forecast_drawdown, actual_drawdown = drawdown_plot(fore_drawdown_smape, act_drawdown_smape, forecast_drawdown, actual_drawdown)



    return render(request, 'Forecast.html', 
                  {'Date': dateToday,
                   'actual': last_known_value,
                   'forecasted': forecasted,
                   'forecasted_date': forecasted_date,
                   'fore_smape': fore_smape,
                   'act_smape': act_smape,
                   'forecast_drawdown': forecast_drawdown,
                   'actual_drawdown': actual_drawdown,
                   'fore_drawdown_smape':fore_drawdown_smape,
                   'act_drawdown_smape': act_drawdown_smape,
                   'actual_rain': actual_rain,
                   'forecast_rain': forecast_rain,
                   'fore_rain_smape': fore_rain_smape,
                   'act_rain_smape': act_rain_smape})

def Business_zone (request):
    import datetime
    from datetime import datetime as dt_time
    data = pd.read_csv('manila_water_data.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
    last_date = data['Date'].iloc[-1]
    alloc_date_format = pd.to_datetime(last_date, format='%d-%b-%y')
    get_month = alloc_date_format.month
    get_year = alloc_date_format.year
    day = 1
    if request.method == 'POST':
        get_month = int(request.POST['month']) 
        get_year = int(request.POST['year'])

    datetime_obj = dt_time(year=get_year,month=get_month, day=day)

    display_year = datetime_obj.strftime("%Y")
    display_month = datetime_obj.strftime("%B")
    display_date = f"{display_month} {display_year}"

    date_string = datetime_obj.strftime("%d-%b-%y")

    filtered_data = data[data['Date'].dt.strftime('%d-%b-%y') == date_string]
    filtered_data.set_index('Business Zone', inplace=True)
    filtered_data.sort_index(inplace=True)
    index = filtered_data.index
    filtered_data['nrwv'] = filtered_data['Supply Volume'] - filtered_data['Bill Volume']
    def water_alloc_plot():
        df = pd.DataFrame(filtered_data, columns=['Supply Volume', 'nrwv'], index=index)
        plt.style.use("dark_background")
        ax = df.plot.bar(stacked=True, rot=0, figsize=(10, 7), color=('Blue', 'skyblue'))
        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy() 
            ax.text(x + width/2, y + height/2, f'{height:.2f}', ha='center', va='center', color='black', fontsize=11)
            
        plt.yticks(color='white', fontsize=13)
        plt.xticks(color='white', fontsize=13)
        plt.legend(loc='upper right', fontsize='smaller', ncol=1, borderpad=1.0, borderaxespad=1.0, shadow=True)
        plt.tight_layout() 
        plt.savefig('decision_support/static/img/bz_water_alloc(1).png', transparent=True)
    water_alloc_plot()

    total_supply = filtered_data['Supply Volume'].sum()
    total_nrwv = filtered_data['nrwv'].sum()
    nrwv_percentage = (total_nrwv / total_supply) * 100
    return render(request, 'Business-Zones.html', 
                  {'Date': dateToday,
                   'total_supply':total_supply,
                   'total_nrwv':total_nrwv,
                   'nrwv_percentage': nrwv_percentage,
                   'display_date':display_date})
    