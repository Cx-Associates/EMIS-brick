"""
Get data from Ace and dents
Combine data
Be prepared to answer any questions that might come up from this
"""
import os
import requests
import pandas as pd
import yaml

#Copied from new_model - sorting through it
import pickle

from src.utils import Project, EnergyModelset, join_csv_data
from config_MSL import config_dict, heating_system_Btus

def parse_response(response):
    """

    :param response:
    :return:
    """
    dict_ = response.json()
    list_ = dict_['point_samples']
    df = pd.DataFrame(list_)
    df.index = pd.to_datetime(df.pop('time'))
    df.drop(columns='name', inplace=True)
    df['value'] = pd.to_numeric(df['value'])

    return df

env_filename = 'api_keys.yml'
f_drive_path = 'F:/PROJECTS/1715 Main Street Landing EMIS Pilot/code/API keys'
env_filepath = os.path.join(f_drive_path, env_filename)
timezone='US/Eastern'

# create an instance of the project class, giving it a name and a location
project = Project(
    name=config_dict['name'],
    location=config_dict['location'],
)

# set the project baseline period
project.set_time_frames(
    baseline=('2023-11-10', '2023-12-18'),
    #reporting=('2023-12-10', '2023-12-18')
)

"""
# set filepath for brick model .ttl file, and load it into the project
graph_path = 'brick_models/msl_heating-cooling.ttl'
project.load_graph(graph_path)

# create an instance of the energy modelset class and designate the systems for which to create individual energy models
modelset = EnergyModelset(
    project,
    systems=[
        'heating_system',
        # 'chilled_water_system',
    ],
    equipment=[
        'chiller'
    ]
)

# get weather data, then get relevant building timeseries data
#project.get_weather_data()
modelset.get_data()

#end
"""
str=[r'/cxa_main_st_landing/2404:9-240409/analogOutput/5/timeseries?start_time=2023-11-10&end_time=2023-12-18',
r'/cxa_main_st_landing/2404:9-240409/analogOutput/6/timeseries?start_time=2023-11-10&end_time=2023-12-18',
r'/cxa_main_st_landing/2404:9-240409/analogInput/15/timeseries?start_time=2023-11-10&end_time=2023-12-18',
r'/cxa_main_st_landing/2404:9-240409/analogInput/16/timeseries?start_time=2023-11-10&end_time=2023-12-18',
r'/cxa_main_st_landing/2404:7-240407/analogValue/11/timeseries?start_time=2023-11-10&end_time=2023-12-18']

#Lets try this instead?
with open(env_filepath, 'r') as file:
    config = yaml.safe_load(file)
    url = config['DATABASE_URL']
    for str_ in str:
        str_ = url + str_
        auth_token = config['API_KEY']
        headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json',
        }
        res = requests.get(str_, headers=headers)
        if res.status_code == 200:
            print(f'...Got data! From: \n {str_} \n')
            df = parse_response(res)
            df.index = df.index.tz_localize('UTC').tz_convert(timezone)
        else:
            msg = f'API request from ACE was unsuccessful. \n {res.reason} \n {res.content}'
            #raise Exception(msg)

    print('hello')
#pull out data I want to compile
#This is going to take some work, and probably needs to happen in the for loop above.

dentdatapath=r"F:/PROJECTS/1715 Main Street Landing EMIS Pilot/data/Dent data pull, 2024-01-05/raw data\\E876C-01.csv"

#join_csv_data(modelset.systems.'heating_system'.dataframe,dentdatapath)
