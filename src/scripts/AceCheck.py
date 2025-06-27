"""
Something funny with Ace?  Use this code to pull some data and see how things look.
Outputs data in this location: "F:\PROJECTS\1715 Main Street Landing EMIS Pilot\data"
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

#Todo: START AND END DATES - EDIT THESE TO GET THE TIME FRAME YOU WANT TO LOOK AT

start = '2025-06-01' #YYYY-MM-DD
end = '2025-06-27' #YYYY-MM-DD

#Ace Stuff
env_filename = 'api_keys.yml'
f_drive_path = 'F:/PROJECTS/1715 Main Street Landing EMIS Pilot/code/API keys'
env_filepath = os.path.join(f_drive_path, env_filename)
timezone='US/Eastern'

ACE_data = pd.DataFrame() #Defining empty dataframe into which BMS data will be pulled into from ACE API


def parse_response(response,columnname):
    """
    This reads JSON responses from AceIOT.
    """
    dict_ = response.json()
    list_ = dict_['point_samples']
    df = pd.DataFrame(list_)
    df.index = pd.to_datetime(df.pop('time'))
    df.drop(columns='name', inplace=True)
    df[columnname] = pd.to_numeric(df['value'])
    df=df.drop(columns='value')

    return df

#Ace Data locations- JUST A FEW FOR FUN
str = [fr'/cxa_main_st_landing/2404:9-240409/analogOutput/5/timeseries?start_time={start}&end_time={end}', #Pump 4a VFD Output
fr'/cxa_main_st_landing/2404:9-240409/analogOutput/6/timeseries?start_time={start}&end_time={end}', #Pump 4b VFD Output
fr'/cxa_main_st_landing/2404:7-240407/analogValue/11/timeseries?start_time={start}&end_time={end}', #chilled water power meter
fr'/cxa_main_st_landing/2404:7-240407/analogOutput/4/timeseries?start_time={start}&end_time={end}', #pump 2a-b VFD output
fr'/cxa_main_st_landing/2404:3-240403/analogOutput/5/timeseries?start_time={start}&end_time={end}', #Exhaust fan 1 VFD speed
fr'/cxa_main_st_landing/2404:3-240403/analogOutput/6/timeseries?start_time={start}&end_time={end}', #Exhaust fan 2 VFD speed
fr'/cxa_main_st_landing/2404:3-240403/analogOutput/2/timeseries?start_time={start}&end_time={end}', #Heat Recovery Wheel VFD
fr'/cxa_main_st_landing/2404:3-240403/binaryInput/6/timeseries?start_time={start}&end_time={end}',
fr'/cxa_main_st_landing/2404:9-240409/analogOutput/3/timeseries?start_time={start}&end_time={end}',
fr'/cxa_main_st_landing/2404:10-240410/analogOutput/8/timeseries?start_time={start}&end_time={end}'
] #Boiler 1% signal] #Heat Recovery Wheel Status

#Ace Data descriptions
headers = ['Pump 4a VFD Output',
         'Pump 4b VFD Output',
         'Chilled water power meter',
         'Pump 2a-b VFD output',
         'AHU19 Exhaust fan 1 VFD speed',
         'AHU19 Exhaust fan 2 VFD speed',
         'AHU19 Heat Recovery Wheel VFD',
         'AHU19 Heat Recovery Wheel Status',
         'Boiler 1% signal',
         'HRU supply fan VFD output'
]

#For each path for AceIoT data (listed above) and the description, get the data and put the data in the data frame with header listed above
with open(env_filepath, 'r') as file:
    config = yaml.safe_load(file)
    url = config['DATABASE_URL']
    for str_, head in zip(str, headers):
        str_ = url + str_
        auth_token = config['API_KEY']
        headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json',
        }
        res = requests.get(str_, headers=headers)
        if res.status_code == 200:
            print(f'...Got data! From: \n {str_} \n')
            df = parse_response(res, head)
            df.index = df.index.tz_localize('UTC').tz_convert(timezone)
            #do 15-minute averages
            #df_15min = df[head].resample('15T').mean()

            ACE_data = pd.DataFrame.merge(ACE_data, df, left_index=True, right_index=True, how='outer')
        else:
            msg = f'API request from ACE was unsuccessful. \n {res.reason} \n {res.content}'
            #raise Exception(msg) #Uncomment this to troubleshoot any points that are not being downloaded


#Export the data for troubleshooting
main_folder = r"F:\PROJECTS\1715 Main Street Landing EMIS Pilot\data"
ACE_data_file_path = os.path.join(main_folder, f'ACE_Data_Check_5min_{end}.csv')
ACE_data.to_csv(ACE_data_file_path)