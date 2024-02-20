"""
Get data from Ace and dents
Combine data
Create some correlation plots
Create some time series plots
"""
import os
import requests
import pandas as pd
import yaml
import matplotlib.pyplot as plt

# Set default plot parameters
plt.rcParams['lines.linestyle'] = ''
plt.rcParams['lines.marker'] = '.'
plt.rcParams['lines.markersize'] = 1

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
    #^for some reason that has us off by 4 hours?
    MSL_data1.set_index('CombinedDatetime', inplace=True)
    MSL_data1.index = MSL_data1.index.tz_localize('US/Eastern', ambiguous='NaT')
    # Handle ambiguous time error by dropping rows with NaT values in the index
    MSL_data1 = MSL_data1[MSL_data1.index.notnull()]
#   MSL_data1.index = MSL_data1.index.tz_localize('UTC').tz_convert(timezone)

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
    baseline=('2023-11-10', '2023-12-31'),
    #reporting=('2023-12-10', '2023-12-18')
)

#Ace Data locations
str = [r'/cxa_main_st_landing/2404:9-240409/analogOutput/5/timeseries?start_time=2023-11-10&end_time=2024-01-06', #Pump 4a VFD Output
r'/cxa_main_st_landing/2404:9-240409/analogOutput/6/timeseries?start_time=2023-11-10&end_time=2024-01-06', #Pump 4b VFD Output
r'/cxa_main_st_landing/2404:9-240409/analogInput/15/timeseries?start_time=2023-11-10&end_time=2024-01-06', #Primary Hot Water Supply Temp_2
r'/cxa_main_st_landing/2404:9-240409/analogInput/16/timeseries?start_time=2023-11-10&end_time=2024-01-06', #Primary Hot Water Return Temp_2
r'/cxa_main_st_landing/2404:7-240407/analogValue/11/timeseries?start_time=2023-11-10&end_time=2024-01-06', #chilled water power meter
r'/cxa_main_st_landing/2404:7-240407/analogOutput/4/timeseries?start_time=2023-11-10&end_time=2024-01-06', #pump 2a-b VFD output
r'/cxa_main_st_landing/2404:7-240407/binaryOutput/12/timeseries?start_time=2023-11-10&end_time=2024-01-06', #pump 2a active (binary)
r'/cxa_main_st_landing/2404:7-240407/binaryOutput/13/timeseries?start_time=2023-11-10&end_time=2024-01-06', #pump 2b active (binary)
r'/cxa_main_st_landing/2404:2-240402/analogInput/10/timeseries?start_time=2023-11-10&end_time=2024-01-06', #P1a Feedback
r'/cxa_main_st_landing/2404:10-240410/analogOutput/8/timeseries?start_time=2023-11-10&end_time=2024-01-06', #HRU Supplyfan VFD output
r'/cxa_main_st_landing/2404:3-240403/analogOutput/3/timeseries?start_time=2023-11-10&end_time=2024-01-06'] #AHU19 Supply fan VFD

#Ace Data descriptions
headers = ['Pump 4a VFD Output',
         'Pump 4b VFD Output',
         'Primary Hot Water Supply Temp_2',
         'Primary Hot Water Return Temp_2',
         'chilled water power meter',
         'pump 2a-b VFD output',
         'pump 2a active',
         'pump 2b active',
         'pump 1a feedback',
         'HRU supply fan',
         'AHU19 supply fan']

#Lets try this
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
            #add the column header to be useful!
            df = parse_response(res, head)
            df.index = df.index.tz_localize('UTC').tz_convert(timezone)
            #do 15-minute averages
            df_15min = df[head].resample('15T').mean()

            MSL_data=pd.merge(df_15min, MSL_data, left_index=True, right_index=True, how='outer')
        else:
            msg = f'API request from ACE was unsuccessful. \n {res.reason} \n {res.content}'
            #raise Exception(msg)

#Lets do some math to prepare for correlations!
MSL_data['pump2a'] = MSL_data['pump 2a active']*MSL_data['pump 2a-b VFD output'] #this should be the BAS pump 2a power (ish)
MSL_data['pump2b'] = MSL_data['pump 2b active']*MSL_data['pump 2a-b VFD output'] #same for pump 2b

#Correlation time!
plt.plot(MSL_data['pump2a'], MSL_data['Avg. Amp Pump 2a'])
plt.xlabel('BAS VFD output')
plt.ylabel('Dent Amp data for Pump 2a')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump2aCorrelation.png')
plt.close()

plt.plot(MSL_data['pump2b'], MSL_data['Avg. Amp Pump 2b'])
plt.xlabel('BAS VFD output')
plt.ylabel('Dent Power data for Pump 2b')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump2bCorrelation.png')
plt.close()

plt.plot(MSL_data['pump 1a feedback'], MSL_data['Avg. Amp Pump 1a'])
plt.xlabel('BAS Pump 1a Feedback')
plt.ylabel('Dent Power data for Pump 1')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump1Correlation.png')
plt.close()

plt.plot(MSL_data['AHU19 supply fan'], MSL_data['Avg. Amp AHU19'])
plt.xlabel('BAS AHU19 Supply Fan')
plt.ylabel('Dent Power data for AHU 19')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\AHU19Correlation.png')
plt.close()

plt.plot(MSL_data['HRU supply fan'], MSL_data['Avg. AmpL1 HRU'])
plt.plot(MSL_data['HRU supply fan'], MSL_data['Avg. AmpL2 HRU'])
plt.xlabel('HRU Supply fan')
plt.ylabel('Dent Power data for HRU')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\HRUCorrelation.png')
plt.close()


plt.plot(MSL_data.index,MSL_data['pump2a'])
plt.plot(MSL_data.index,MSL_data['Avg. Amp Pump 2a'])
plt.legend(['Ace Data','Dent Data'])
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump2aTimeserries.png')
plt.close()

plt.plot(MSL_data.index,MSL_data['pump2b'])
plt.plot(MSL_data.index,MSL_data['Avg. Amp Pump 2b'])
plt.legend(['Ace Data','Dent Data'])
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump2bTimeserries.png')
plt.close()

plt.plot(MSL_data.index,MSL_data['pump 1a feedback'])
plt.plot(MSL_data.index,MSL_data['Avg. Amp Pump 1a'])
plt.legend(['Ace Data (Pump 1a feedback)','Dent Data'])
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump1aTimeserries.png')
plt.close()

plt.plot(MSL_data.index,MSL_data['AHU19 supply fan'])
plt.plot(MSL_data.index,MSL_data['Avg. Amp AHU19'])
plt.legend(['Ace Data (AHU19 Supply Fan)','Dent Data (all AHU19)'])
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\AHU19Timeserries.png')
plt.close()

plt.plot(MSL_data.index,MSL_data['HRU supply fan'])
plt.plot(MSL_data.index,MSL_data['Avg. AmpL1 HRU'])
plt.plot(MSL_data.index,MSL_data['Avg. AmpL2 HRU'])
plt.legend(['Ace Data (HRU Supply Fan)','Dent Data Phase 1','Dent Data Phase 2'])
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\HRUTimeserries.png')
plt.close()

MSL_data.plot(y=['pump 2a-b VFD output',
         'pump 2a active',
         'pump 2b active',
         'Avg. Amp Pump 2a',
         'Avg. Amp Pump 2b'])
plt.show()

