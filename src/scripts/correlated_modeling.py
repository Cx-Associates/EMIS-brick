"""
HELP - what does this code do!!
SOME CALCS FOR MONTHLY REPORTING
"""

import os
import requests
import pandas as pd
import yaml
from datetime import date
from dateutil.relativedelta import relativedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry
import calendar
import math
import numpy as np
from datetime import datetime
from pathlib import Path
from dateutil import tz
from sklearn.linear_model import LinearRegression
import builtins

start_baseline_heating = '2024-11-01'#"xx-xx-xxxx" #start of baseline period #todo: update when baseline period is determined, current dates are for heating system baseline
end_baseline_heating = '2025-02-01'#"xx-xx-xxxx" #end of baseline period #todo: update when baseline period is determined, current dates are for heating system baseline
start_baseline_vent = '2024-11-01'#"xx-xx-xxxx" #start of baseline period for ventilation #todo: update when baseline period is determined, current dates are for ventilation system baseline
end_baseline_vent = '2025-03-01'#"xx-xx-xxxx" #end of baseline period for ventilation #todo: update when baseline period is determined, current dates are for ventilation system baseline

#from src.utils import Project
#from config_MSL import config_dict

#Import Correlation Parameters
corr_path = "F:/PROJECTS/1715 Main Street Landing EMIS Pilot/code/RegressionParameters.csv"
Corr_param_df = pd.DataFrame(pd.read_csv(corr_path))
baseline_corr_path = r"F:/PROJECTS/1715 Main Street Landing EMIS Pilot/code/Baseline_Model_Regression_Parameters.csv" #Contains just heating system baseline currently
baseline_corr_df = pd.DataFrame(pd.read_csv(baseline_corr_path))


def parse_response(response,columnname):
    """
    This reads JSON responses from AceIOT.
    """
    dict_ = response.json()
    alist_ = dict_['point_samples']
    df = pd.DataFrame(alist_)
    df.index = pd.to_datetime(df.pop('time'))
    df.drop(columns='name', inplace=True)
    df[columnname] = pd.to_numeric(df['value'])
    df=df.drop(columns='value')

    return df

env_filename = 'api_keys.yml'
f_drive_path = 'F:/PROJECTS/1715 Main Street Landing EMIS Pilot/code/API keys'
env_filepath = os.path.join(f_drive_path, env_filename)
timezone='US/Eastern'

#Define reporting period
today = datetime.today() #use this format if you need to force it a previous month datetime(2024,10,25)
a_month_ago = today - relativedelta(months=1) #Setting monthly reporting period
start = a_month_ago.replace(day=1) # Get the first day of the previous month
last_day_of_prev_month = calendar.monthrange(a_month_ago.year, a_month_ago.month)[1] # Get the last day of the previous month
end = today.replace(day=4) #Setting it a bit ahead because not seeing complete data from ACE api otherwise
end_rep = today.replace(day=last_day_of_prev_month).strftime('%Y-%m-%d')
end_rep = str(end_rep)
end_check = datetime(a_month_ago.year, a_month_ago.month, hour = 23, minute =0, second = 0)
start = str(start.strftime('%Y-%m-%d'))
end = str(end) #Start and end dates need to be strings

#Create datetime varibales to drop unnecessary rows
start_check = pd.to_datetime(start).tz_localize(timezone)
end_check = pd.to_datetime(end_check).tz_localize(timezone)

# Mapping EDT and EST to America/New_York handles both DST and standard time
tzinfos = {
    'EDT': tz.gettz('US/Eastern'),
    'EST': tz.gettz('US/Eastern'),
}

ACE_data = pd.DataFrame() #Defining empty dataframe into which BMS data will be pulled into from ACE API

def get_value(equipment_name, data): #Todo: For next project it will be good to define all functions in the utils.py or a seperate one
    index = data['Equipt'].index(equipment_name)  # Find index of equipment name
    size = data['value'][index]  # Retrieve corresponding size using index
    return size

def get_hp(equipment_name, data):
    index = data['Equipt'].index(equipment_name)  # Find index of equipment name
    size = data['hp'][index]  # Retrieve corresponding size using index
    return size

#Ace Data locations
mystr = [fr'/cxa_main_st_landing/2404:9-240409/analogOutput/5/timeseries?start_time={start}&end_time={end}', #Pump 4a VFD Output
fr'/cxa_main_st_landing/2404:9-240409/analogOutput/6/timeseries?start_time={start}&end_time={end}', #Pump 4b VFD Output
#fr'/cxa_main_st_landing/2404:9-240409/binaryOutput/12/timeseries?start_time={start}&end_time={end}', #Pump 4a s/s
#fr'/cxa_main_st_landing/2404:9-240409/binaryOutput/13/timeseries?start_time={start}&end_time={end}', #Pump 4b s/s
#fr'/cxa_main_st_landing/2404:9-240409/analogInput/16/timeseries?start_time={start}&end_time={end}', #Primary Hot Water Supply Temp_2
#fr'/cxa_main_st_landing/2404:9-240409/analogInput/15/timeseries?start_time={start}&end_time={end}', #Primary Hot Water Return Temp_2
fr'/cxa_main_st_landing/2404:9-240409/analogOutput/3/timeseries?start_time={start}&end_time={end}', #Boiler 1% signal
fr'/cxa_main_st_landing/2404:9-240409/analogOutput/4/timeseries?start_time={start}&end_time={end}', #Boiler 2% signal
fr'/cxa_main_st_landing/2404:9-240409/binaryOutput/6/timeseries?start_time={start}&end_time={end}', #Boiler 1 Status
fr'/cxa_main_st_landing/2404:9-240409/binaryOutput/7/timeseries?start_time={start}&end_time={end}', #Boiler 2 Status'
fr'/cxa_main_st_landing/2404:7-240407/analogValue/11/timeseries?start_time={start}&end_time={end}', #chilled water power meter
fr'/cxa_main_st_landing/2404:7-240407/analogOutput/4/timeseries?start_time={start}&end_time={end}', #pump 2a-b VFD output
fr'/cxa_main_st_landing/2404:7-240407/binaryInput/18/timeseries?start_time={start}&end_time={end}', #Pump 2a status
fr'/cxa_main_st_landing/2404:7-240407/binaryInput/19/timeseries?start_time={start}&end_time={end}', #Pump 2b status
fr'/cxa_main_st_landing/2404:2-240402/analogInput/10/timeseries?start_time={start}&end_time={end}', #P1a Feedback
fr'/cxa_main_st_landing/2404:2-240402/analogInput/11/timeseries?start_time={start}&end_time={end}', #P1b Feedback
fr'/cxa_main_st_landing/2404:2-240402/analogOutput/3/timeseries?start_time={start}&end_time={end}', #P1 VFD Signal
fr'/cxa_main_st_landing/2404:7-240407/binaryInput/9/timeseries?start_time={start}&end_time={end}', #Pump 3a status
fr'/cxa_main_st_landing/2404:7-240407/binaryInput/10/timeseries?start_time={start}&end_time={end}', #Pump 3b status
fr'/cxa_main_st_landing/2404:7-240407/binaryOutput/5/timeseries?start_time={start}&end_time={end}', #Chiller Status (binary)
fr'/cxa_main_st_landing/2404:7-240407/analogInput/8/timeseries?start_time={start}&end_time={end}', #Chiller HX1 Flow (GPM) (only flow data we have for chiller)
fr'/cxa_main_st_landing/2404:7-240407/analogInput/21/timeseries?start_time={start}&end_time={end}', #Chilled water supply temp (F)
fr'/cxa_main_st_landing/2404:7-240407/analogInput/20/timeseries?start_time={start}&end_time={end}', #Chilled water return temp (F)
fr'/cxa_main_st_landing/2404:7-240407/analogInput/17/timeseries?start_time={start}&end_time={end}', #Condenser Water Supply Temperature (F)
fr'/cxa_main_st_landing/2404:7-240407/analogInput/13/timeseries?start_time={start}&end_time={end}', #Condenser Water Return Temperature (F)
fr'/cxa_main_st_landing/2404:2-240402/analogInput/7/timeseries?start_time={start}&end_time={end}', #Cooling Tower Temp In (F)
fr'/cxa_main_st_landing/2404:2-240402/analogInput/8/timeseries?start_time={start}&end_time={end}', #Cooling Tower Temp Out (F)
fr'/cxa_main_st_landing/2404:7-240407/binaryValue/11/timeseries?start_time={start}&end_time={end}', #Cooling Tower Free Cool Status (binary)
fr'/cxa_main_st_landing/2404:2-240402/analogOutput/4/timeseries?start_time={start}&end_time={end}', #Cooling tower fan %speed
fr'/cxa_main_st_landing/2404:2-240402/binaryInput/10/timeseries?start_time={start}&end_time={end}', #Cooling tower Fan1 Status
fr'/cxa_main_st_landing/2404:2-240402/binaryInput/11/timeseries?start_time={start}&end_time={end}', #Cooling tower Fan2 Status
fr'/cxa_main_st_landing/2404:10-240410/analogOutput/8/timeseries?start_time={start}&end_time={end}', #HRU Supply fan VFD output
fr'/cxa_main_st_landing/2404:10-240410/analogOutput/2/timeseries?start_time={start}&end_time={end}', #HRU Exhaust Fan VFD speed
fr'/cxa_main_st_landing/2404:10-240410/binaryInput/1/timeseries?start_time={start}&end_time={end}', #HRU Exhaust Fan Status
fr'/cxa_main_st_landing/2404:10-240410/binaryInput/9/timeseries?start_time={start}&end_time={end}', #HRU Supply Fan Status
fr'/cxa_main_st_landing/2404:3-240403/analogOutput/3/timeseries?start_time={start}&end_time={end}', #AHU19 Supply fan VFD
fr'/cxa_main_st_landing/2404:3-240403/binaryInput/3/timeseries?start_time={start}&end_time={end}', #AHU19 Supply fan Status
fr'/cxa_main_st_landing/2404:3-240403/analogOutput/5/timeseries?start_time={start}&end_time={end}', #Exhaust fan 1 VFD speed
fr'/cxa_main_st_landing/2404:3-240403/analogOutput/6/timeseries?start_time={start}&end_time={end}', #Exhaust fan 2 VFD speed
fr'/cxa_main_st_landing/2404:3-240403/analogOutput/2/timeseries?start_time={start}&end_time={end}', #Heat Recovery Wheel VFD
fr'/cxa_main_st_landing/2404:3-240403/binaryInput/6/timeseries?start_time={start}&end_time={end}', #Heat Recovery Wheel Status
fr'/cxa_main_st_landing/2404:3-240403/analogValue/9/timeseries?start_time={start}&end_time={end}', #Exhaust fan CFM
fr'/cxa_main_st_landing/2404:3-240403/analogValue/16/timeseries?start_time={start}&end_time={end}', #Total Cool Request from Zones
fr'/cxa_main_st_landing/2404:3-240403/analogValue/17/timeseries?start_time={start}&end_time={end}', #Total Heat Request from Zones
fr'/cxa_main_st_landing/2404:9-240409/binaryInput/19/timeseries?start_time={start}&end_time={end}', #P4B Status
fr'/cxa_main_st_landing/2404:9-240409/binaryInput/18/timeseries?start_time={start}&end_time={end}'] #P4A Status


#Ace Data descriptions
headers = ['Pump 4a VFD Output',
         'Pump 4b VFD Output',
#         'Pump 4a s/s',
#         'Pump 4b s/s',
#         'Primary Hot Water Supply Temp_2',
#         'Primary Hot Water Return Temp_2',
         'Boiler 1% signal',
         'Boiler 2% signal',
         'Boiler 1 status',
         'Boiler 2 status',
         'Chilled water power meter',
         'Pump 2a-b VFD output',
         'Pump 2a status',
         'Pump 2b status',
         'Pump 1a feedback',
         'Pump 1b feedback',
         'Pump 1 VFD Signal',
         'Pump 3a status',
         'Pump 3b status',
         'Chiller status',
         'Chiller HX1 Flow (GPM)',
         'Chilled water supply temp (F)',
         'Chilled water return temp (F)',
         'Condenser Water Supply Temperature (F)',
         'Condenser Water Return Temperature (F)',
         'Cooling Tower Temp In (F)',
         'Cooling Tower Temp Out (F)',
         'Cooling Tower Free Cool Status',
         'Cooling tower fan %speed',
         'Cooling tower Fan 1 Status',
         'Cooling tower Fan 2 Status',
         'HRU supply fan VFD output',
         'HRU Exhaust fan VFD output',
         'HRU Exhaust Fan Status',
         'HRU Supply Fan Status',
         'AHU19 supply fan VFD output',
         'AHU19 Supply fan Status',
         'AHU19 Exhaust fan 1 VFD speed',
         'AHU19 Exhaust fan 2 VFD speed',
         'AHU19 Heat Recovery Wheel VFD',
         'AHU19 Heat Recovery Wheel Status',
         'AHU19 Exhaust fan CFM',
         'Total Cool Request from Zones',
         'Total Heat Request from Zones',
         'Pump 4b Status',
         'Pump 4a Status']

#For each path for AceIoT data (listed above) and the description, get the data and put the data in the data frame with header listed above
with open(env_filepath, 'r') as file:
    config = yaml.safe_load(file)
    url = config['DATABASE_URL']
    for mystr_, head in zip(mystr, headers):
        mystr_ = url + mystr_
        auth_token = config['API_KEY']
        headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json',
        }
        res = requests.get(mystr_, headers=headers)
        if res.status_code == 200:
            print(f'...Got data! From: \n {mystr_} \n')
            df = parse_response(res, head)
            df.index = df.index.tz_localize('UTC').tz_convert(timezone)
            #do 15-minute averages
            #df_15min = df[head].resample('15T').mean()

            ACE_data = pd.DataFrame.merge(ACE_data, df, left_index=True, right_index=True, how='outer')
        else:
            msg = f'API request from ACE was unsuccessful. \n {res.reason} \n {res.content}'
            #raise Exception(msg) #Uncomment this to troubleshoot any points that are not being downloaded

#READ IN BMS DATA
#get paths for all the files
BMS_path =Path(r"F:\PROJECTS\1715 Main Street Landing EMIS Pilot\Monthly Reports\Progress Report_2025-03-31\BMS Data")
csv_files = list(BMS_path.glob('**/*.csv'))
BASdatapath = [str(file.resolve()) for file in csv_files]

#read in BAS data (for where we didn't get appropriate data from Ace):
for path in BASdatapath:
    # Read in data from a file with data collected via direct download from the BAS
    BAS_data1 = pd.read_csv(path, skiprows=1, header=0)
    folder_name = os.path.basename(os.path.dirname(path))
    BAS_data1[folder_name] = BAS_data1['Value']
    BAS_data1['Date'] = BAS_data1['Date'].str.replace('EDT', '')
    BAS_data1['Date'] = BAS_data1['Date'].str.replace('EST', '')
    BAS_data1['CombinedDatetime'] = pd.to_datetime(BAS_data1['Date'])
    BAS_data1.set_index('CombinedDatetime', inplace=True)
    BAS_data1.index = BAS_data1.index.tz_localize('US/Eastern', ambiguous='NaT',
                                                  nonexistent='shift_forward')  # Non-existent deals with daylight savings
    BAS_data1 = BAS_data1.loc[(BAS_data1.index >= start)]
    if 'BAS_data' in locals():
        BAS_data = pd.merge(BAS_data, BAS_data1[folder_name], left_index=True, right_index=True, how='outer')
    else:
        BAS_data=BAS_data1

#Deal with change of value data
BAS_data.sort_index(inplace=True)
BAS_data = BAS_data[~BAS_data.index.duplicated()]
BAS_data.replace(['nan', 'NaN', '', 'None', ' nan', 'nan '], pd.NA,inplace=True)
BAS_data=BAS_data.resample('1min').ffill()
BAS_data.ffill(inplace=True)
BAS_data=BAS_data.resample('5min').mean()

#Combine ACE data and BAS data with preference for ACE data.
ACE_data=ACE_data.combine_first(BAS_data)

#Create folder to store data and report
main_folder = r"F:\PROJECTS\1715 Main Street Landing EMIS Pilot\Monthly Reports"
subfolder_name = f"Progress Report_{end_rep}"
subfolder_path = os.path.join(main_folder, subfolder_name)
os.makedirs(subfolder_path, exist_ok=True) # Create the subfolder if it doesn't exist
ACE_data_file_path = os.path.join(subfolder_path, f'ACE_Data_5min_{end_rep}.csv')
ACE_data.to_csv(ACE_data_file_path) #Uncomment this out when the start and end dates have changed or any change in data is expected. This will write over the existing file.

#Pump/fan nameplates
Nameplate= {'Equipt':['Pump1a', 'Pump1b', 'Pump2a', 'Pump2b', 'Pump4a', 'Pump4b', 'HRUSupplyFan', 'HRUReturnFan',
                        'AHU19SupplyFan', 'AHU19ReturnFan', 'Pump3a', 'Pump3b', "CTFan1", "CTFan2", "AHU19EF1", "AHU19EF2","AHU19SF", "AHU19HRW"], 'hp':[20, 15, 25, 25, 7.5, 7.5, 10, 10, 7.5, 10, 7.5, 7.5, 15, 15, 10, 10, 7.5, 0.1]}
nameplate=pd.DataFrame(Nameplate)

#Boiler Nameplate
Boiler_Nameplate = {'Equipt':['Boiler1_capacity', 'Boiler2_capacity', 'Boiler1_Eff', 'Boiler2_Eff'], 'value':[2047, 2081, 0.878, 0.891]} #Boiler capacity is in MBtu/hr

##Normalization
balance_point_HDD = 45 #Updated on 03/03/25
balance_point_CDD = 65 #These base temp will be calculated once we have enough data to establish a baseline/balance point. These values are taken from AHSRAE recommendation: https://www.ashrae.org/File%20Library/Technical%20Resources/Building%20Energy%20Quotient/User-Tip-5_May2019.pdf

#Get weather data from Open Meteo #Todo: For future projects convert this into a function that just takes the start and end date as inputs. Probably the variables too?

#Setup the Open-Meteo API client with cache and retry on error.
cache_session = requests_cache.CachedSession('.cache', expire_after=3600) #Caching prevents the need for multiple API calls which is important since open meteo has a fixed number of free API calls
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

#Define the parameters as variables
latitude = 44.48
longitude = -73.21
hourly_variables = ["temperature_2m", "dew_point_2m", "precipitation", "weather_code"]
temperature_unit = "fahrenheit"
timezone = "America/New_York"
start_date = start
end_date = end_rep

#Parameters dictionary using variables
params = {
    "latitude": latitude,
    "longitude": longitude,
    "hourly": hourly_variables,
    "temperature_unit": temperature_unit,
    "timezone": timezone,
    "start_date": start_date,
    "end_date": end_date
}

#Make the API request
responses = openmeteo.weather_api(url="https://api.open-meteo.com/v1/forecast", params=params)

# Process the first location. Add a loop if needed for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

#Process hourly data. Ensure the order of variables matches the request
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
hourly_weather_code = hourly.Variables(3).ValuesAsNumpy()

#Prepare hourly data with date range and assign the values
hourly_data = {
    "date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ),
    "temperature_2m": hourly_temperature_2m,
    "dew_point_2m": hourly_dew_point_2m,
    "precipitation": hourly_precipitation,
    "weather_code": hourly_weather_code
}

#Create a DataFrame from the hourly data
hourly_weather_dataframe = pd.DataFrame(data=hourly_data)
#hourly_weather_dataframe = hourly_weather_dataframe[(hourly_weather_dataframe.index>=start_check) & (hourly_weather_dataframe.index<=end_check)]

#Calculate HDD (when temperature is below balance_point_HDD)
hourly_weather_dataframe['HDD'] = hourly_weather_dataframe['temperature_2m'].apply(
    lambda temp: abs(temp - balance_point_HDD) if temp < balance_point_HDD else 0)

#Calculate CDD (when temperature is above balance_point_CDD)
hourly_weather_dataframe['CDD'] = hourly_weather_dataframe['temperature_2m'].apply(
    lambda temp: abs(temp - balance_point_CDD) if temp > balance_point_CDD else 0)

#Output the DataFrame
#hourly_weather_dataframe.to_csv("Open_meteo_weather_data.csv") #Todo: Comment this out before push

#Calculate the daily total HDD and CDD for each hours #Todo: Needs to be removed possibly
hourly_weather_df = hourly_weather_dataframe.drop(['dew_point_2m', 'precipitation', 'weather_code'], axis=1) #Dropping whatever variables are not going to be important
hourly_weather_df['date'] = pd.to_datetime(hourly_weather_df['date']) #todo: Add timezone, and that might fix it? #Date from open meteo is a RangeIndex and resample only works on DatetimeIndex, TimedeltaIndex, or PeriodIndex. The dt.date drops the time component otherwise resampling was not working properly.
hourly_weather_df.set_index('date', inplace=True)#Setting the date as index
hourly_weather_df.index = hourly_weather_df.index.tz_convert('US/Eastern')
#hourly_weather_df.to_csv('Hourly_weather_df.csv') #You know the drill

#make 5-minute data
hourly_weather_df = hourly_weather_df[~hourly_weather_df.index.duplicated()]
hourly_weather_df.replace(['nan', 'NaN', '', 'None', ' nan', 'nan '], pd.NA,inplace=True)
weather_df=hourly_weather_df.resample('5min').ffill()
weather_df.ffill(inplace=True)

#add temp data to ACE_data
ACE_data = pd.merge(ACE_data, weather_df, how='outer', left_index=True, right_index=True)


#Create the final dataframe which will be used for graphing
Report_df = pd.DataFrame() #Dataframe which will store all calculated energy consumption and any data needed for reporting

#Todo: For future projects, will be good if we do system level correlations instead of doing equipment level, will reduce steps and problems with loss of data
#Todo: For future projects, we should have different files for each system so that if data is missing for certain points, it doesn't break the whole code. Also easier troubleshooting.

##HEATING SYSTEM CALCS
#Create system level dataframes #Todo: For future projects, will be good to create system level functions which will take in equipments as input and use that to calculate system level energy consumption
Heating_df = ACE_data[['Pump 4a VFD Output', 'Pump 4b VFD Output', 'Boiler 1% signal', 'Boiler 2% signal']] #Always use double [] brackets for picking the data you need
columns_to_fill = ['Pump 4a VFD Output', 'Pump 4b VFD Output', 'Boiler 1% signal', 'Boiler 2% signal']
#Heating_df.to_csv('Heating_df.csv') #Uncomment for troubleshooting

#for missing data use similar temperature times to fill the missing data:
for col in columns_to_fill:
    missing_indices = Heating_df[Heating_df[col].isna()].index
    for idx in missing_indices:
        temp_at_time = Heating_df.loc[idx, 'temperature_2m']

        if pd.isna(temp_at_time):
            # Skip if temp itself is missing
            continue

        # Find rows within +/- 3 degrees of the temp at this missing point
        temp_mask = Heating_df['temperature_2m'].between(temp_at_time - 3, temp_at_time + 3)
        valid_rows = Heating_df[temp_mask & df[col].notna()]

        if len(valid_rows) < 2:
            # Not enough data to build a model, widen search:
            temp_mask = Heating_df['temperature_2m'].between(temp_at_time - 8, temp_at_time + 8)
            valid_rows = Heating_df[temp_mask & df[col].notna()]
            if len(valid_rows) < 2:
                continue #just give up...

        # Fit linear model: temperature_2m -> col
        X = valid_rows[['temperature_2m']]
        y = valid_rows[col]
        model = LinearRegression().fit(X, y)

        # Predict using temperature at the missing index
        predicted_value = model.predict(np.array([[temp_at_time]]))[0]

        # Fill in the missing value
        df.loc[idx, col] = predicted_value

#Calculating kW from BMS information #Todo: For a future project try to get rid of the warning:  value is trying to be set on a copy of a slice from a DataFrame.
Heating_df['Pump 4a kW (Formula Based)'] = get_hp('Pump4a',Nameplate)*0.745699872*(Heating_df['Pump 4a VFD Output']/100)**2.5
Heating_df['Pump 4b kW (Formula Based)'] = get_hp('Pump4b',Nameplate)*0.745699872*(Heating_df['Pump 4b VFD Output']/100)**2.5
Heating_df['Boiler 1 MBtu'] = get_value('Boiler1_capacity', Boiler_Nameplate)*Heating_df['Boiler 1% signal']/(100*get_value('Boiler1_Eff', Boiler_Nameplate)) #The 100 in the denominator is to convert the % signal value
Heating_df['Boiler 2 MBtu'] = get_value('Boiler2_capacity', Boiler_Nameplate)*Heating_df['Boiler 2% signal']/(100*get_value('Boiler2_Eff', Boiler_Nameplate))
#Heating_df.to_csv('Heating_df.csv') #Uncomment for troubleshooting
Heating_df_15min = Heating_df.resample(rule='15Min').mean() #Averaging for 15 min in order to use the correlation factors
#Heating_df_15min.to_csv('Heating_df_15min.csv') #Uncomment for troubleshooting

#Calculating correlated values and adding reporting variables to dataframe
Report_df['Boiler 1 MBtu'] = Heating_df_15min['Boiler 1 MBtu']
Report_df['Boiler 2 MBtu'] = Heating_df_15min['Boiler 2 MBtu']
Report_df['Total Boiler NG Consumption (MBtu)'] = Heating_df_15min[['Boiler 1 MBtu', 'Boiler 2 MBtu']].sum(axis=1, min_count=1)
Report_df['Pump 4a kW (Correlated)'] = Heating_df_15min['Pump 4a kW (Formula Based)'] * Corr_param_df['slope'][0] + Corr_param_df['intercept'][0]
Report_df['Pump 4b kW (Correlated)'] = Heating_df_15min['Pump 4b kW (Formula Based)'] * Corr_param_df['slope'][1] + Corr_param_df['intercept'][1]
Report_df['Heating System kW'] = Report_df[['Pump 4a kW (Correlated)', 'Pump 4b kW (Correlated)']].sum(axis=1, min_count=1) #If there are columns with NAN data then the sum won't work and the total column will also be nan. The min_count deals with this so if at least one column has a non-NaN value, the sum will be computed using the non-NaN values. The NaN values will be ignored

#Report_df.to_csv('Report_df.csv') #Uncomment for troubleshooting

##AHU-19 CALCS
AHU_df = ACE_data[['AHU19 supply fan VFD output', 'AHU19 Exhaust fan 1 VFD speed', 'AHU19 Exhaust fan 2 VFD speed', 'AHU19 Heat Recovery Wheel VFD']]

#Calculating kW from BMS information
AHU_df['AHU 19 EF1 kW (Formula Based)'] = (get_hp('AHU19EF1', Nameplate))*0.745699872*(AHU_df['AHU19 Exhaust fan 1 VFD speed']/100)**2.5 #No status exists
AHU_df['AHU 19 EF2 kW (Formula Based)'] = (get_hp('AHU19EF2', Nameplate))*0.745699872*(AHU_df['AHU19 Exhaust fan 2 VFD speed']/100)**2.5 #No status exists
AHU_df['AHU 19 SF kW (Formula Based)'] = (get_hp('AHU19SF', Nameplate))*0.745699872*(AHU_df['AHU19 supply fan VFD output']/100)**2.5
AHU_df['AHU 19 HRW kW (Formula Based)'] = (get_hp('AHU19HRW', Nameplate))*0.745699872*(AHU_df['AHU19 Heat Recovery Wheel VFD']/100)**2.5
AHU_df['AHU 19 Total kW (Formula Based)'] = AHU_df[['AHU 19 EF1 kW (Formula Based)', 'AHU 19 EF2 kW (Formula Based)', 'AHU 19 SF kW (Formula Based)', 'AHU 19 HRW kW (Formula Based)']].sum(axis=1, min_count=1)
AHU_df_15min = AHU_df.resample(rule='15Min').mean()
#AHU_df.to_csv('AHU_df.csv) #Uncomment for troubleshooting

#Calculating correlated values and adding reporting variables to dataframe
Report_df['AHU 19 Total kW (Correlated)'] = AHU_df_15min['AHU 19 Total kW (Formula Based)']* Corr_param_df['slope'][5] + Corr_param_df['intercept'][5]

##HRU CALCS
HRU_df = ACE_data[['HRU supply fan VFD output', 'HRU Exhaust fan VFD output']]
HRU_df['HRU Supply Fan kW (Formula Based)'] = (get_hp('HRUSupplyFan',Nameplate))*0.745699872*(HRU_df['HRU supply fan VFD output']/100)**2.5
HRU_df['HRU Exhaust Fan kW (Formula Based)'] = (get_hp('HRUReturnFan',Nameplate))*0.745699872*(HRU_df['HRU Exhaust fan VFD output']/100)**2.5
HRU_df['HRU Total kW (Formula Based)'] = HRU_df[['HRU Exhaust Fan kW (Formula Based)', 'HRU Supply Fan kW (Formula Based)']].sum(axis=1, min_count=1) #One DENT on all HRU so will need to correlate to total
HRU_df_15min = HRU_df.resample(rule='15Min').mean()
#HRU_df_15min.to_csv('HRU_df_15min.csv')

#Calculating correlated values and adding reporting variables to dataframe
Report_df['HRU Total kW (Correlated)'] = HRU_df_15min['HRU Total kW (Formula Based)'] * Corr_param_df['slope'][4] + Corr_param_df['intercept'][4]

#CHILLED WATER SYSTEM CALCS
CHW_df = ACE_data[[ 'Pump 2a-b VFD output', 'Pump 2a status', 'Pump 2b status', 'Pump 3a status', 'Pump 3b status', 'Pump 1a feedback', 'Pump 1b feedback', 'Pump 1 VFD Signal', 'Cooling tower fan %speed', 'Cooling tower Fan 1 Status', 'Cooling tower Fan 2 Status']]
chiller_df=ACE_data[['Chilled water power meter','Chiller status',]] #doing this one separate since it has shorter data storage in the BMS

#Calculating kW from BMS information
CHW_df['Pump 1a kW (Formula Based)'] = get_hp('Pump1a',Nameplate)*0.745699872*(CHW_df['Pump 1a feedback']/100)**2.5 #No status exists #todo: add correlation if needed
CHW_df['Pump 1b kW (Formula Based)'] = get_hp('Pump1b', Nameplate)*0.745699872*(CHW_df['Pump 1b feedback']/100)**2.5 #No status exists and does not need correlation
CHW_df['Pump 2b kW (Formula Based)'] = CHW_df['Pump 2a status']*get_hp('Pump2b',Nameplate)*0.745699872*(CHW_df['Pump 2a-b VFD output']/100)**2.5
CHW_df['Pump 2a kW (Formula Based)'] = CHW_df['Pump 2b status']*(get_hp('Pump2a',Nameplate)*0.745699872*(CHW_df['Pump 2a-b VFD output']/100)**2.5)
CHW_df['Pump 3a kW (Formula Based)'] = CHW_df['Pump 3a status']*(get_hp('Pump3a', Nameplate))*0.745699872
CHW_df['Pump 3b kW (Formula Based)'] = CHW_df['Pump 3b status']*(get_hp('Pump3a', Nameplate))*0.745699872
CHW_df['Tower Fan 1 kW (Formula Based)'] = CHW_df['Cooling tower Fan 1 Status']*(get_hp('CTFan1', Nameplate))*0.745699872*(CHW_df['Cooling tower fan %speed']/100)**2.5
CHW_df['Tower Fan 2 kW (Formula Based)'] = CHW_df['Cooling tower Fan 2 Status']*(get_hp('CTFan2', Nameplate))*0.745699872*(CHW_df['Cooling tower fan %speed']/100)**2.5


CHW_df['Chiller kW'] = chiller_df['Chiller status'] * chiller_df['Chilled water power meter']

CHW_df_15min = CHW_df.resample(rule='15Min').mean()
#CHW_df_15min.to_csv("CHW_df_15min.csv")

#Calculating correlated values and adding reporting variables to dataframe
#Report_df['Pump 1a kW (Correlated)'] = CHW_df_15min['Pump 1a kW (Formula Based)']* Corr_param_df['slope'][x] + Corr_param_df['intercept'][x] #Todo: Add corr parameters when available
Report_df['Pump 1a kW (Formula Based)'] = CHW_df_15min['Pump 1a kW (Formula Based)']
Report_df['Pump 1b kW (Formula Based)'] = CHW_df_15min['Pump 1b kW (Formula Based)']
Report_df['Pump 2a kW (Correlated)'] = CHW_df_15min['Pump 2a kW (Formula Based)'] * Corr_param_df['slope'][2] + Corr_param_df['intercept'][2]
Report_df['Pump 2b kW (Correlated)'] = CHW_df_15min['Pump 2b kW (Formula Based)'] * Corr_param_df['slope'][3] + Corr_param_df['intercept'][3]
Report_df['Pump 3a kW (Formula Based)'] = CHW_df_15min['Pump 3a kW (Formula Based)']
Report_df['Pump 3b kW (Formula Based)'] = CHW_df_15min['Pump 3b kW (Formula Based)']
Report_df['Tower Fan 1 kW (Correlated)'] = CHW_df_15min['Tower Fan 1 kW (Formula Based)'] * Corr_param_df['slope'][6] + Corr_param_df['intercept'][6] #the only electricty consuming equipment in the CTs are the fan assemblies
Report_df['Tower Fan 2 kW (Correlated)'] = CHW_df_15min['Tower Fan 2 kW (Formula Based)'] * Corr_param_df['slope'][7] + Corr_param_df['intercept'][7]
Report_df['Chiller kW'] = CHW_df_15min['Chiller kW']
Report_df['Total CHW kW'] = Report_df[['Pump 1a kW (Formula Based)', 'Pump 1b kW (Formula Based)', 'Pump 2a kW (Correlated)', 'Pump 2b kW (Correlated)', 'Pump 3a kW (Formula Based)', 'Pump 3b kW (Formula Based)', 'Tower Fan 1 kW (Correlated)', 'Tower Fan 2 kW (Correlated)', 'Chiller kW']].sum(axis=1, min_count=1)

#Report_df.to_csv('Report_df_15min.csv')
Report_df_hourly = Report_df.resample(rule='H').mean() #Resmpling and aggregating consumption hourly. After this step all electric values are essentially kWh since we get the kW consumed during that hour.
#Report_df_hourly.to_csv('Report_df_hourly.csv') #You know the drill


Report_df_final['Total Boiler NG Consumption (MMBtu)'] = Report_df_final['Total Boiler NG Consumption (MBtu)']/1000
Report_df_final['Total Heating Plant Energy Consumption (MMBtu)'] = Report_df_final['Total Boiler NG Consumption (MMBtu)'] + (Report_df_final['Heating System kW'] * 0.003412) #converting total consumption to MMBtu

# List of columns to check for NaN values. Due to difference in how open meteo and ACE handle API requests, we get some additional rows where we have no ACE data
columns_to_check = ['Total Heating Plant Energy Consumption (MMBtu)', 'AHU 19 Total kW (Correlated)',
                    'HRU Total kW (Correlated)', 'Total CHW kW']

#Drop rows where all the specified columns have NaN values
Report_df_final= Report_df_final.dropna(subset=columns_to_check, how='all')
Report_df_final.index = pd.to_datetime(Report_df_final.index)

#Check to make sure data is just within the reporting period
Report_df_final = Report_df_final[(Report_df_final.index>= start_check) & (Report_df_final.index <=end_check)]

#TODO: MOVE THIS FILLING IN TO THE INDIVIDUAL DATA FRAMES, SO THAT MORE DATA CAN BE INCLUDED AND WE CAN GET EACH PIECE OF EQUIPMENT
#If some data is missing in the final reporting data frame fill it in as appropriate
#first, check how much data we expect
#number of expected hours:
diff = end_check - start_check
exp_hours = diff.total_seconds()/3600 +2
#second, check how much data we actually have for each column in the dataframe

#find nans in the reporting data frame
nans_per_column = Report_df_final.isna().sum()
columns_with_nans = nans_per_column[nans_per_column > 0]

#fill in nans with appropriate data:
#for meteo data: use average of preceding and following values, and flag if more than 4 hours!
meteocolumns = ['HDD', 'CDD','temperature_2m']
def has_consecutive_nans(series, n=4):
    return (series.isna()
                  .astype(int)
                  .groupby(series.notna().cumsum())
                  .sum()
                  .gt(n)
                  .any())

for col in meteocolumns:
    if Report_df_final[col].isna().any():
        Report_df_final[col] = Report_df_final[col].interpolate(method='linear', limit_direction='both') #average before and after values
        if has_consecutive_nans(Report_df_final[col], n=4):
            print(f"Column '{col}' has more than 4 NaNs in a row.")

#for ventilation system, similar day/time values from other parts of the month:

Report_df_final['weekday'] = Report_df_final.index.dt.dayofweek     # Monday=0, Sunday=6
Report_df_final['time'] = Report_df_final.index.dt.time              # Time part only

# Set of columns where you want this fill strategy
ventcolumns = ['AHU 19 Total kW (Correlated)', 'HRU Total kW (Correlated)']

for col in ventcolumns:
    if Report_df_final[col].isna().any():
        # Create a helper key to group on weekday and time
        Report_df_final['group_key'] = list(zip(Report_df_final['weekday'], Report_df_final['time']))

        # Compute mean values for each (weekday, time) group
        group_means = Report_df_final.groupby('group_key')[col].transform('mean')

        # Fill NaNs with the group mean
        Report_df_final[col] = Report_df_final[col].fillna(group_means)

# Cleanup (optional)
df.drop(columns=['weekday', 'time', 'group_key'], inplace=True)






#Report_df_final.to_csv(f"Report_df_final_{end}.csv")

##Write the final dataframe to the F drive
file_path = os.path.join(subfolder_path, f"Report_df_final_{end_rep}.csv")
Report_df_final.to_csv(file_path)

#Total energy calculations
Total_energy_MMBtu = format(round((
        Report_df_final['Total Heating Plant Energy Consumption (MMBtu)'].sum() +
        (Report_df_final['AHU 19 Total kW (Correlated)'].sum()* 0.003412) +
        (Report_df_final['HRU Total kW (Correlated)'].sum()* 0.003412) +
        (Report_df_final['Total CHW kW'].sum()* 0.003412)), 1
),",")

total_energy_system_level = pd.DataFrame({
    'Ventilation': [(Report_df_final['AHU 19 Total kW (Correlated)'].sum() + Report_df_final['HRU Total kW (Correlated)'].sum()) * 0.003412],
    'Chilled Water System': [Report_df_final['Total CHW kW'].sum() * 0.003412],
    'Heating Plant': [Report_df_final['Total Heating Plant Energy Consumption (MMBtu)'].sum()]
})

csv_file_path = os.path.join(subfolder_path, f'Total_energy_system_level_{end_rep}.csv')
total_energy_system_level.to_csv(csv_file_path)

#Write total energy into a csv file for historical data trend graph
csv_file_path = r"F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Total_Energy_MMBtu_History.csv"
start_date = datetime.strptime(start, "%Y-%m-%d")
month_year = start_date.strftime("%B %Y")  #Get the month name and year (e.g., "August 2024")
new_data = pd.DataFrame({"Month-Year": [month_year], "Total Energy (MMBtu)": [Total_energy_MMBtu]})
energy_history_df = pd.read_csv(csv_file_path)
if not ((energy_history_df['Month-Year'] == month_year).any()): #If the month-year is not in the dataframe, append the new row
    energy_history_df = pd.concat([energy_history_df, new_data], ignore_index=True)
energy_history_df.to_csv(csv_file_path, index=False)

##Calculate NG Usage in CCF and Electricity in kWh
NG_Usage_CCF = format(round(Report_df_final['Total Boiler NG Consumption (MMBtu)'].sum() / .1026),",") #https://portfoliomanager.energystar.gov/pdf/reference/Thermal%20Conversions.pdf
Electricty_Usage_kWh = format(round(Report_df_final['Heating System kW'].sum() + Report_df_final['Total CHW kW'].sum() + Report_df_final['AHU 19 Total kW (Correlated)'].sum() + Report_df_final['HRU Total kW (Correlated)'].sum()),",")
#print(NG_Usage_CCF)
#print(Electricty_Usage_kWh)

#Todo: All normalization needs to be done based on today's (09/25/24) discussion between RH and LB. We first establish a baseline equaltion so first step is determiniing a balance point, second is use the balance point to calculate HDD and CDD, the fit  a trendline for the baseline case, our predicted actual energy consumption will be using this equation with the actual DD. We will also plot the "actual" energy consumption.
#Todo: Add normalization below

#Normalizing the energy consumption using baseline #Todo: Once baseline is established, graphs needs to be updated to display the expected (normalized) energy consumption and graphs need to be updated to add the black bars. All of the below calcs are not being utilized currently.
total_energy_system_corr = pd.DataFrame()

#{
#     'AHU19': [Report_df_final['AHU 19 Total kW (Correlated)'].sum()],
#     'HRU': [Report_df_final['HRU Total kW (Correlated)'].sum()],
#     'Chilled Water System': [Report_df_final['Total CHW kW'].sum()],
#     'Heating Plant': [Report_df_final['Total Heating Plant Energy Consumption (MMBtu)'].sum()]
# })
#
total_energy_system_corr['Heating Plant (Normalized)'] = Report_df_final['HDD'] * baseline_corr_df['slope'][0] + baseline_corr_df['intercept'][0]
# total_energy_system_corr['AHU19 (Normalized)'] = total_energy_system_corr['AHU19'] * baseline_corr_df['slope'][1] + baseline_corr_df['intercept'][1]
# total_energy_system_corr['HRU (Normalized)'] = total_energy_system_corr['HRU'] * baseline_corr_df['slope'][2] + baseline_corr_df['intercept'][2]
# total_energy_system_corr['Chilled Water System (Normalized)'] = total_energy_system_corr['Chilled Water System'] * baseline_corr_df['slope'][3] + baseline_corr_df['intercept'][3]

#This outputs the necessary information for the reporting
#Report Period Start Date
startd = datetime.strptime(start,"%Y-%m-%d")
startdateformated=startd.strftime("%B %d, %Y")
STARTdate=startd.strftime("%B %d, %Y").upper()
Month=startd.strftime("%B %Y")
#Report Period End Date
endd = datetime.strptime(end_rep,"%Y-%m-%d")
enddateformated=endd.strftime("%B %d, %Y")
ENDdate=endd.strftime("%B %d, %Y").upper()
#baseline times
sbhn = datetime.strptime(start_baseline_heating,"%Y-%m-%d")
start_baseline_heatingnice=sbhn.strftime("%m/%d/%Y")
ebhn =  datetime.strptime(end_baseline_heating,"%Y-%m-%d")
end_baseline_heatingnice = ebhn.strftime("%m/%d/%Y")


TeXfolderpath = subfolder_path.replace("\\", "/")

texoutput_file_path = os.path.join(subfolder_path, f'Output.tex')
with open(texoutput_file_path,'w') as tex_file:
    # Write the variable to the file
    tex_file.write(f"\\newcommand{{\\Figpath}}{{{TeXfolderpath}}}\n")
    tex_file.write(f"\\newcommand{{\\Month}}{{{Month}}}\n")
    tex_file.write(f"\\newcommand{{\\StartDate}}{{{startdateformated}}}\n")
    tex_file.write(f"\\newcommand{{\\StartDateCap}}{{{STARTdate}}}\n")
    tex_file.write(f"\\newcommand{{\\EndDate}}{{{enddateformated}}}\n")
    tex_file.write(f"\\newcommand{{\\EndDateCap}}{{{ENDdate}}}\n")
    tex_file.write(f"\\newcommand{{\\TotalEnergy}}{{{Total_energy_MMBtu}}}\n")
    tex_file.write(f"\\newcommand{{\\TotalCCF}}{{{NG_Usage_CCF}}}\n")
    tex_file.write(f"\\newcommand{{\\TotalElectricity}}{{{Electricty_Usage_kWh}}}\n")
    tex_file.write(f"\\newcommand{{\\heatingbaselinestart}}{{{start_baseline_heatingnice}}}\n") #todo actually use the right variable here down!
    tex_file.write(f"\\newcommand{{\\heatingbaselineend}}{{{end_baseline_heatingnice}}}\n")