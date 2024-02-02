"""
Get data from Ace and dents
Combine data
Be prepared to answer any questions that might come up from this
"""
import os
import requests
import pandas as pd
import yaml
import matplotlib.pyplot as plt

#Copied from new_model - sorting through it
import pickle

from src.utils import Project, EnergyModelset
from config_MSL import config_dict, heating_system_Btus

def parse_response(response,columnname):
    """
    :param response:
    :return:
    """
    dict_ = response.json()
    list_ = dict_['point_samples']
    df = pd.DataFrame(list_)
    df.index = pd.to_datetime(df.pop('time'))
    df.drop(columns='name', inplace=True)
    df[columnname] = pd.to_numeric(df['value'])
    df=df.drop(columns='value')

    return df

#some places and specifics
dentdatapath=[r"F:/PROJECTS/1715 Main Street Landing EMIS Pilot/data/Dent data pull, 2024-01-05/raw data/E876C-01.csv",
              r"F:/PROJECTS/1715 Main Street Landing EMIS Pilot/data/Dent data pull, 2024-01-05/raw data/E876D-01.csv",
              r"F:/PROJECTS/1715 Main Street Landing EMIS Pilot/data/Dent data pull, 2024-01-05/raw data/E876F-01.csv"]
env_filename = 'api_keys.yml'
f_drive_path = 'F:/PROJECTS/1715 Main Street Landing EMIS Pilot/code/API keys'
env_filepath = os.path.join(f_drive_path, env_filename)
timezone='US/Eastern'


for path in dentdatapath:
    # Read in data from a file with data collected via on-site monitoring
    MSL_data1 = pd.read_csv(path, skiprows=1, header=0)

    # for dent data they have new lines in them sometimes, let's remove those
    MSL_data1.columns = MSL_data1.columns.str.replace('\n', '')
    MSL_data1.columns = MSL_data1.columns.str.replace('\r', '')

    # create 'time' column
    MSL_data1['CombinedDatetime']=pd.to_datetime(MSL_data1['Date '] + ' ' + MSL_data1['End Time '])
    MSL_data1.set_index('CombinedDatetime', inplace=True)
    MSL_data1.index = MSL_data1.index.tz_localize('UTC').tz_convert(timezone)
    # Combine with existing data frame
    if 'MSL_data' in locals():
        MSL_data = pd.merge(MSL_data, MSL_data1, left_index=True, right_index=True, how='outer')
    else:
        MSL_data=MSL_data1



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
str=[r'/cxa_main_st_landing/2404:9-240409/analogOutput/5/timeseries?start_time=2023-11-10&end_time=2023-12-31', #Pump 4a VFD Output
r'/cxa_main_st_landing/2404:9-240409/analogOutput/6/timeseries?start_time=2023-11-10&end_time=2023-12-31', #Pump 4b VFD Output
r'/cxa_main_st_landing/2404:9-240409/analogInput/15/timeseries?start_time=2023-11-10&end_time=2023-12-31', #Primary Hot Water Supply Temp_2
r'/cxa_main_st_landing/2404:9-240409/analogInput/16/timeseries?start_time=2023-11-10&end_time=2023-12-31', #Primary Hot Water Return Temp_2
r'/cxa_main_st_landing/2404:7-240407/analogValue/11/timeseries?start_time=2023-11-10&end_time=2023-12-31', #chilled water power meter
r'/cxa_main_st_landing/2404:7-240407/analogOutput/4/timeseries?start_time=2023-11-10&end_time=2023-12-31', #pump 2a-b VFD output
r'cxa/cxa_main_st_landing/2404:7-240407/binaryOutput/12/timeseries?start_time=2023-11-10&end_time=2023-12-31', #pump 2a active (binary)
r'cxa/cxa_main_st_landing/2404:7-240407/binaryOutput/13/timeseries?start_time=2023-11-10&end_time=2023-12-31'] #pump 2b active (binary)

headers=['Pump 4a VFD Output',
         'Pump 4b VFD Output',
         'Primary Hot Water Supply Temp_2',
         'Primary Hot Water Return Temp_2',
         'chilled water power meter',
         'pump 2a-b VFD output',
         'pump 2a active',
         'pump 2b active']

#Lets try this
#For each path for AceIoT data (listed above), get the data and put the data in the data frame
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
            # somehow add the column header to be useful!
            df = parse_response(res, head)
            df.index = df.index.tz_localize('UTC').tz_convert(timezone)

            MSL_data=pd.merge(df, MSL_data, left_index=True, right_index=True, how='outer')
        else:
            msg = f'API request from ACE was unsuccessful. \n {res.reason} \n {res.content}'
            #raise Exception(msg)

MSL_data.plot(y=['pump 2a-b VFD output',
         'pump 2a active',
         'pump 2b active',
         'Avg. AmpL1 Phase_x',
         'Avg. AmpL2 Phase_x'])
plt.show()

