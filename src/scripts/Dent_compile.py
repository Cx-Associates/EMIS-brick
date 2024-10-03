"""
Get data from Ace and dents (and BAS where missing Ace data)
Combine data
Calculate estimated kW using proxy formulas
Create some correlation plots and correlation values (and save)
Create some time series plots (and save)

Outputs results in file "RegressionParameters"
Lots of plots are also saved in the plots folder

The DENT that is labeled as being on Pump 1a is actually on Pump 2b - so note that this all comes out as Pump 2b stuff

"""

import os
import requests
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import csv


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

def get_hp(equipment_name, data):
    index = data['Equipt'].index(equipment_name)  # Find index of equipment name
    size = data['hp'][index]  # Retrieve corresponding size using index
    return size


#some places and specifics #todo: This data is all compiled from multiple data pulls, pd.merge doesn't do well with similar column headers. Future work to update code to accept similar column headers over multiple time periods
dentdatapath=[r"F:/PROJECTS/1715 Main Street Landing EMIS Pilot/data/Dent data pull, 2024-01-05/raw data/E876C-01.csv",
              r"F:/PROJECTS/1715 Main Street Landing EMIS Pilot/data/Dent data pull, 2024-01-05/raw data/E876D-01.csv",
              r"F:/PROJECTS/1715 Main Street Landing EMIS Pilot/data/Dent data pull, 2024-01-05/raw data/E876F-01.csv",
              r"F:/PROJECTS/1715 Main Street Landing EMIS Pilot/data/Dent data pull, 2024-08-20/XC1409243-01_CT.csv"]

#Ace gateway data is missing between "2024-04-24 22:10:00-04:00" and "2024-07-15 14:15:00-04:00" (additional points were added a few days later so some points missing here and there)

BASdatapath=[r"F:\PROJECTS\1715 Main Street Landing EMIS Pilot\data\BAS Trend Data\Chilled Water System\BAS TrendData CoolingTower combined.csv",
             r"F:\PROJECTS\1715 Main Street Landing EMIS Pilot\data\BAS Trend Data\Chilled Water System\BAS CT1 Status combined.csv",
             r"F:\PROJECTS\1715 Main Street Landing EMIS Pilot\data\BAS Trend Data\Chilled Water System\BAS CT2 Status combined.csv",
             r"F:\PROJECTS\1715 Main Street Landing EMIS Pilot\data\BAS Trend Data\AHU19\Heat Wheel VFD combined.csv",
             r"F:\PROJECTS\1715 Main Street Landing EMIS Pilot\data\BAS Trend Data\AHU19\EF1 Signal combined.csv",
             r"F:\PROJECTS\1715 Main Street Landing EMIS Pilot\data\BAS Trend Data\AHU19\EF2 Signal combined.csv",
             r"F:\PROJECTS\1715 Main Street Landing EMIS Pilot\data\BAS Trend Data\AHU19\SF VFD Speed combined.csv",
             r"F:\PROJECTS\1715 Main Street Landing EMIS Pilot\data\BAS Trend Data\HRU\EF1 Signal combined.csv",
             r"F:\PROJECTS\1715 Main Street Landing EMIS Pilot\data\BAS Trend Data\HRU\SF Signal combined.csv"]

env_filename = 'api_keys.yml'
f_drive_path = 'F:/PROJECTS/1715 Main Street Landing EMIS Pilot/code/API keys'
env_filepath = os.path.join(f_drive_path, env_filename)
timezone='US/Eastern'

#Get dent data into a dataframe lined up in time
for path in dentdatapath:
    # Read in data from a file with data collected via on-site monitoring
    MSL_data1 = pd.read_csv(path, skiprows=1, header=0)

    # for dent data they have new lines in them sometimes, let's remove those
    MSL_data1.columns = MSL_data1.columns.str.replace('\n', '')
    MSL_data1.columns = MSL_data1.columns.str.replace('\r', '')
    MSL_data1.columns = MSL_data1.columns.str.strip() #Was getting a key error 'Date ', this resolved the issue so most likely it was to do with blank spaces.

    # create 'time' column
    MSL_data1['CombinedDatetime']=pd.to_datetime(MSL_data1['Date'] + ' ' + MSL_data1['End Time'])
    #^for some reason that has us off by 4 hours?
    MSL_data1.set_index('CombinedDatetime', inplace=True)
    MSL_data1.index = MSL_data1.index.tz_localize('US/Eastern', ambiguous='NaT', nonexistent='shift_forward') #Non-existent deals with daylight savings
    # Handle ambiguous time error by dropping rows with NaT values in the index
    MSL_data1 = MSL_data1[MSL_data1.index.notnull()]

    # Combine with existing data frame
    if 'MSL_data' in locals():
        MSL_data = pd.merge(MSL_data, MSL_data1, left_index=True, right_index=True, how='outer')
        #MSL_data = pd.join(MSL_data, MSL_data1)
    else:
        MSL_data=MSL_data1

#read in BAS data (for where we didn't get appropriate data from Ace):
for path in BASdatapath:
    # Read in data from a file with data collected via on-site monitoring
    BAS_data1 = pd.read_csv(path, skiprows=1, header=0)
    BAS_data1['CombinedDatetime'] = pd.to_datetime(BAS_data1['Date'])
    BAS_data1.set_index('CombinedDatetime', inplace=True)
    BAS_data1.index = BAS_data1.index.tz_localize('US/Eastern', ambiguous='NaT',
                                                  nonexistent='shift_forward')  # Non-existent deals with daylight savings
    if 'BAS_data' in locals():
        BAS_data = pd.merge(BAS_data, BAS_data1, left_index=True, right_index=True, how='outer')
    else:
        BAS_data=BAS_data1

# create an instance of the project class, giving it a name and a location
project = Project(
    name=config_dict['name'],
    location=config_dict['location'],
)
# Define the gateway down and back up timeframes, dates were determined from ACE_data_5
#gateway_down = pd.Timestamp("2024-04-24 22:10:00-04:00", tz='US/Eastern')
#gateway_up = pd.Timestamp("2024-07-15 14:15:00-04:00", tz='US/Eastern')

# Filter out the rows between gateway_down and gateway_up
#MSL_data = MSL_data[(MSL_data.index < gateway_down) | (MSL_data.index > gateway_up)]


#Pump/fan nameplates
Nameplate= {'Equipt':['Pump1a', 'Pump1b', 'Pump2a', 'Pump2b', 'Pump4a', 'Pump4b', 'HRUSupplyFan', 'HRUReturnFan',
                        'AHU19SupplyFan', 'AHU19ReturnFan', 'Pump3a', 'Pump3b', "CTFan1", "CTFan2", "AHU19EF1", "AHU19EF2","AHU19SF", "AHU19HRW"], 'hp':[20, 15, 25, 25, 7.5, 7.5, 10, 10, 7.5, 10, 7.5, 7.5, 15, 15, 10, 10, 7.5, 0.1]}

start = "2023-09-20"
end = "2024-09-01"

#Ace Data locations
#Todo: make this a file you pull from instead of hard coded.
str = [fr'/cxa_main_st_landing/2404:9-240409/analogOutput/5/timeseries?start_time={start}&end_time={end}', #Pump 4a VFD Output
fr'/cxa_main_st_landing/2404:9-240409/analogOutput/6/timeseries?start_time={start}&end_time={end}', #Pump 4b VFD Output
#fr'/cxa_main_st_landing/2404:9-240409/analogInput/16/timeseries?start_time={start}&end_time={end}', #Primary Hot Water Supply Temp_2
#fr'/cxa_main_st_landing/2404:9-240409/analogInput/15/timeseries?start_time={start}&end_time={end}', #Primary Hot Water Return Temp_2
#fr'/cxa_main_st_landing/2404:9-240409/analogOutput/3/timeseries?start_time={start}&end_time={end}', #Boiler 1% signal
#fr'/cxa_main_st_landing/2404:9-240409/analogOutput/4/timeseries?start_time={start}&end_time={end}', #Boiler 2% signal
#fr'/cxa_main_st_landing/2404:9-240409/binaryOutput/6/timeseries?start_time={start}&end_time={end}', #Boiler 1 Status
#fr'/cxa_main_st_landing/2404:9-240409/binaryOutput/7/timeseries?start_time={start}&end_time={end}', #Boiler 2 Status'
#fr'/cxa_main_st_landing/2404:7-240407/analogValue/11/timeseries?start_time={start}&end_time={end}', #chilled water power meter
fr'/cxa_main_st_landing/2404:7-240407/analogOutput/4/timeseries?start_time={start}&end_time={end}', #pump 2a-b VFD output
fr'/cxa_main_st_landing/2404:7-240407/binaryOutput/12/timeseries?start_time={start}&end_time={end}', #pump 2a activity (binary)
fr'/cxa_main_st_landing/2404:7-240407/binaryOutput/13/timeseries?start_time={start}&end_time={end}', #pump 2b activity (binary)
fr'/cxa_main_st_landing/2404:2-240402/analogInput/10/timeseries?start_time={start}&end_time={end}', #P1a Feedback
fr'/cxa_main_st_landing/2404:2-240402/analogInput/11/timeseries?start_time={start}&end_time={end}', #P1b Feedback
fr'/cxa_main_st_landing/2404:2-240402/analogOutput/3/timeseries?start_time={start}&end_time={end}', #P1 VFD Signal - THIS ONE DOESN'T COME THROUGH
fr'/cxa_main_st_landing/2404:7-240407/binaryInput/9/timeseries?start_time={start}&end_time={end}', #Pump 3a status (binary)
fr'/cxa_main_st_landing/2404:7-240407/binaryInput/10/timeseries?start_time={start}&end_time={end}', #Pump 3b status (binary)
#fr'/cxa_main_st_landing/2404:7-240407/binaryOutput/5/timeseries?start_time={start}&end_time={end}', #Chiller Status (binary)
#fr'/cxa_main_st_landing/2404:7-240407/analogInput/8/timeseries?start_time={start}&end_time={end}', #Chiller HX1 Flow (GPM) (only flow data we have for chiller)
#fr'/cxa_main_st_landing/2404:7-240407/analogInput/21/timeseries?start_time={start}&end_time={end}', #Chilled water supply temp (F)
#fr'/cxa_main_st_landing/2404:7-240407/analogInput/20/timeseries?start_time={start}&end_time={end}', #Chilled water return temp (F)
#fr'/cxa_main_st_landing/2404:7-240407/analogInput/17/timeseries?start_time={start}&end_time={end}', #Condenser Water Supply Temperature (F)
#fr'/cxa_main_st_landing/2404:7-240407/analogInput/13/timeseries?start_time={start}&end_time={end}', #Condenser Water Return Temperature (F)
#fr'/cxa_main_st_landing/2404:2-240402/analogInput/7/timeseries?start_time={start}&end_time={end}', #Cooling Tower Temp In (F) #THIS ONE DOESN't COME THROUGH
#fr'/cxa_main_st_landing/2404:2-240402/analogInput/8/timeseries?start_time={start}&end_time={end}', #Cooling Tower Temp Out (F)
#fr'/cxa_main_st_landing/2404:7-240407/binaryValue/11/timeseries?start_time={start}&end_time={end}', #Cooling Tower Free Cool Status (binary)
fr'/cxa_main_st_landing/2404:2-240402/analogOutput/4/timeseries?start_time={start}&end_time={end}', #Cooling tower fan %speed #THIS ONE DOESN't COME THROUGH
fr'/cxa_main_st_landing/2404:2-240402/binaryInput/10/timeseries?start_time={start}&end_time={end}', #Cooling tower Fan1 Status
fr'/cxa_main_st_landing/2404:2-240402/binaryInput/11/timeseries?start_time={start}&end_time={end}', #Cooling tower Fan2 Status
fr'/cxa_main_st_landing/2404:10-240410/analogOutput/8/timeseries?start_time={start}&end_time={end}', #HRU Supplyfan VFD output
fr'/cxa_main_st_landing/2404:10-240410/analogOutput/2/timeseries?start_time={start}&end_time={end}', #HRU Exhaust Fan VFD speed #THIS ONE DOESN't COME THROUGH
fr'/cxa_main_st_landing/2404:10-240410/binaryInput/1/timeseries?start_time={start}&end_time={end}', #HRU Exhaust Fan Status
fr'/cxa_main_st_landing/2404:10-240410/binaryInput/9/timeseries?start_time={start}&end_time={end}', #HRU Supply Fan Status
fr'/cxa_main_st_landing/2404:3-240403/analogOutput/3/timeseries?start_time={start}&end_time={end}', #AHU19 Supply fan VFD
fr'/cxa_main_st_landing/2404:3-240403/analogOutput/5/timeseries?start_time={start}&end_time={end}', #AHU19 Exhaust fan 1 VFD speed #THIS ONE DOESN't COME THROUGH
fr'/cxa_main_st_landing/2404:3-240403/analogOutput/6/timeseries?start_time={start}&end_time={end}', #AHU19 Exhaust fan 2 VFD speed #THIS ONE DOESN't COME THROUGH
fr'/cxa_main_st_landing/2404:3-240403/analogOutput/2/timeseries?start_time={start}&end_time={end}', #AHU Heat Recovery Wheel VFD #THIS ONE DOESN't COME THROUGH
fr'/cxa_main_st_landing/2404:3-240403/binaryInput/6/timeseries?start_time={start}&end_time={end}', #Heat Recovery Wheel Status #THIS ONE DOESN't COME THROUGH
fr'/cxa_main_st_landing/2404:3-240403/analogValue/9/timeseries?start_time={start}&end_time={end}', #Exhaust fan CFM
# fr'/cxa_main_st_landing/2404:3-240403/analogValue/16/timeseries?start_time={start}&end_time={end}', #Total Cool Request from Zones
# fr'/cxa_main_st_landing/2404:3-240403/analogValue/17/timeseries?start_time={start}&end_time={end}', #Total Heat Request from Zones
fr'/cxa_main_st_landing/2404:7-240407/binaryInput/18/timeseries?start_time={start}&end_time={end}', #Pump 2a status
fr'/cxa_main_st_landing/2404:7-240407/binaryInput/19/timeseries?start_time={start}&end_time={end}',#Pump 2b status
fr'/cxa_main_st_landing/2404:9-240409/binaryOutput/12/timeseries?start_time={start}&end_time={end}',#Pump 4a s/s
fr'/cxa_main_st_landing/2404:9-240409/binaryOutput/13/timeseries?start_time={start}&end_time={end}',#Pump 4b s/s
fr'/ca_main_st_landing/2404:10-240410/binaryValue/5/timeseries?start_time={start}&end_time={end}'] #Occupancy Status

#Ace Data descriptions #Todo: Add statuses when available
headers = ['Pump 4a VFD Output',
         'Pump 4b VFD Output',
         #'Primary Hot Water Supply Temp_2',
         #'Primary Hot Water Return Temp_2',
         #'Boiler 1% signal',
         #'Boiler 2% signal',
         #'Boiler 1 status',
         #'Boiler 2 status',
         #'Chilled water power meter',
         'Pump 2a-b VFD output',
         'Pump 2a activity',
         'Pump 2b activity',
         'Pump 1a feedback',
         'Pump 1b feedback',
         'Pump 1 VFD Signal',
         'Pump 3a status',
         'Pump 3b status',
         # 'Chiller status',
         # 'Chiller HX1 Flow (GPM)',
         # 'Chilled water supply temp (F)',
         # 'Chilled water return temp (F)',
         # 'Condenser Water Supply Temperature (F)',
         # 'Condenser Water Return Temperature (F)',
         # 'Cooling Tower Temp In (F)',
         # 'Cooling Tower Temp Out (F)',
         # 'Cooling Tower Free Cool Status',
         'Cooling tower fan %speed',
         'Cooling tower Fan 1 Status',
         'Cooling tower Fan 2 Status',
         'HRU supply fan VFD output',
         'HRU Exhaust fan VFD output',
         'HRU Exhaust Fan Status',
         'HRU Supply Fan Status',
         'AHU19 supply fan VFD output',
         'AHU19 Exhaust fan 1 VFD speed',
         'AHU19 Exhaust fan 2 VFD speed',
         'AHU19 Heat Recovery Wheel VFD',
         'AHU19 Heat Recovery Wheel Status',
         'AHU19 Exhaust fan CFM',
         # 'Total Cool Request from Zones',
         # 'Total Heat Request from Zones',
         'Pump 2a status',
         'Pump 2b status',
         'Pump 4a s/s',
         'Pump 4b s/s',
           'Occ Status']

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
            #df_15min = df[head].resample('15T').mean()
            if 'Ace_data' in locals():
                Ace_data = pd.merge(df, Ace_data, left_index=True, right_index=True, how='outer')
            else:
                Ace_data = df
        else:
            msg = f'API request from ACE was unsuccessful. \n {res.reason} \n {res.content}'
            #raise Exception(msg)

#Lets do some math to prepare for correlations!
Ace_data['pump2a'] = Ace_data['Pump 2a-b VFD output']*Ace_data['Pump 2a status'] #this should be the BAS pump 2a VFD Output
Ace_data['pump2b'] = Ace_data['Pump 2a-b VFD output']*Ace_data['Pump 2b status'] #same for pump 2b
# Ace_data['CT1'] = Ace_data['Cooling tower fan %speed']*Ace_data['Cooling tower Fan 1 Status'] #same for cooling tower 1
# Ace_data['CT2'] = Ace_data['Cooling tower fan %speed']*Ace_data['Cooling tower Fan 2 Status'] #same for cooling tower 2
BAS_data['CT1'] = BAS_data['BAS CT VFD']*BAS_data['BAS CT1 Status']
BAS_data['CT2'] = BAS_data['BAS CT VFD']*BAS_data['BAS CT2 Status']

#Calculate kW from dent data
#Assume a PF of 0.8 for now:
PF=0.8
MSL_data['Avg. kW Pump 4a'] = MSL_data['Avg. Volt Pump 4a']*MSL_data['Avg. Amp Pump 4a']*3**.5*PF/1000
MSL_data['Avg. kW Pump 4b'] = MSL_data['Avg. Volt Pump 4b']*MSL_data['Avg. Amp Pump 4b']*3**.5*PF/1000
MSL_data['Avg. kW AHU19'] = MSL_data['Avg. Volt AHU19']*MSL_data['Avg. Amp AHU19']*3**.5*PF/1000
MSL_data['Avg. kW Pump 2b'] = MSL_data['Avg. Volt Pump 1a']*MSL_data['Avg. Amp Pump 1a']*3**.5*PF/1000
MSL_data['Avg. kW unknown'] = MSL_data['Avg. Volt Pump 2b']*MSL_data['Avg. Amp Pump 2b']*3**.5*PF/1000
MSL_data['Avg. kW Pump 2a'] = MSL_data['Avg. Volt Pump 2a']*MSL_data['Avg. Amp Pump 2a']*3**.5*PF/1000
MSL_data['Avg. kW HRU'] = (MSL_data['Avg. VoltL1 HRU']*MSL_data['Avg. AmpL1 HRU']*3**.5*PF/1000 + MSL_data['Avg. VoltL2 HRU']*MSL_data['Avg. AmpL2 HRU']*3**.5*PF/1000)/2
MSL_data['Avg. kW AHU9'] = MSL_data['Avg. Volt AHU9']*MSL_data['Avg. Amp AHU9']*3**.5*PF/1000
MSL_data['Avg. kW CT1'] = (MSL_data['Avg. Volt L1 CT1']*MSL_data['Avg. Amp L1 CT1']*3**.5*PF/1000 +MSL_data['Avg. Volt L3 CT1']*MSL_data['Avg. Amp L3 CT1']*3**.5*PF/1000)/2
MSL_data['Avg. kW CT2'] = MSL_data['Avg. Volt L2 CT2']*MSL_data['Avg. Amp L2 CT2']*3**.5*PF/1000

#Calculate expected power from AceData
#=pump nameplate HP *0.745699872*%pump speed^2.5
Ace_data['Ace kW Pump 4a']=get_hp('Pump4a',Nameplate)*0.745699872*(Ace_data['Pump 4a VFD Output']/100)**2.5*Ace_data['Pump 4a s/s']
Ace_data['Ace kW Pump 4b']=get_hp('Pump4b',Nameplate)*0.745699872*(Ace_data['Pump 4b VFD Output']/100)**2.5*Ace_data['Pump 4b s/s']
Ace_data['Ace kW Pump 1a']=get_hp('Pump1a',Nameplate)*0.745699872*(Ace_data['Pump 1a feedback']/100)**2.5 #todo: change this to command!! once we have data!
Ace_data['Ace kW Pump 2b']=get_hp('Pump2b',Nameplate)*0.745699872*(Ace_data['pump2b']/100)**2.5
Ace_data['Ace kW Pump 2a']=(get_hp('Pump2a',Nameplate)*0.745699872*(Ace_data['pump2a']/100)**2.5) #todo: figure out what is happening with HRU exhaust fan data!
Ace_data['HRU kW'] = get_hp('HRUSupplyFan', Nameplate)*0.745699872*(Ace_data['HRU supply fan VFD output']/100)**2.5 #+ get_hp('HRUReturnFan', Nameplate)*0.745699872*(Ace_data['HRU Exhaust fan VFD output']/100)**2.5
#Ace_data['Ace kW AHU19'] = get_hp('AHU19EF2', Nameplate)*0.745699872*(Ace_data['AHU19 Exhaust fan 2 VFD speed']/100)**2.5 + get_hp('AHU19SF', Nameplate)*0.745699872*(Ace_data['AHU19 supply fan VFD output']/100)**2.5 + get_hp('AHU19HRW', Nameplate)*0.745699872*(Ace_data['AHU19 Heat Recovery Wheel VFD']/100)**2.5 #+get_hp('AHU19EF1', Nameplate)*0.745699872*(Ace_data['AHU19 Exhaust fan 1 VFD speed']/100)**2.5 +
# Ace_data['Ace kW CT1']=get_hp('CTFan1',Nameplate)*0.745699872*(Ace_data['CT1']/100)**2.5
# Ace_data['Ace kW CT2']=get_hp('CTFan2',Nameplate)*0.745699872*(Ace_data['CT2']/100)**2.5
BAS_data['BAS kW CT1']=get_hp('CTFan1',Nameplate)*0.745699872*(BAS_data['CT1']/100)**2.5
BAS_data['BAS kW CT2']=get_hp('CTFan2',Nameplate)*0.745699872*(BAS_data['CT2']/100)**2.5
BAS_data['BAS kW AHU19'] = (get_hp('AHU19EF2', Nameplate)*0.745699872*(BAS_data['BAS EF2 Signal']/100)**2.5 +
                            get_hp('AHU19SF', Nameplate)*0.745699872*(BAS_data['BAS SF VFD']/100)**2.5 +
                            get_hp('AHU19HRW', Nameplate)*0.745699872*(BAS_data['AHU HeatWheel VFD']/100)**2.5 +
                            get_hp('AHU19EF1', Nameplate)*0.745699872*(BAS_data['BAS EF1 Signal']/100)**2.5)
BAS_data['BAS kW HRU'] = (get_hp('HRUSupplyFan', Nameplate)*0.745699872*(BAS_data['HRU SF VFD']/100)**2.5 +
                          get_hp('HRUReturnFan', Nameplate)*0.745699872*(BAS_data['HRU EF Signal']/100)**2.5)

#15 minute Ace data averages
Ace_15min = Ace_data.resample(rule='15Min').mean()
BAS_15min = BAS_data.resample(rule='15min').mean()

#comine ace and Dent (MSL) data
MSL_data = pd.merge(MSL_data, Ace_15min, left_index=True, right_index=True, how='outer')
MSL_data = pd.merge(MSL_data, BAS_15min, left_index=True, right_index=True, how='outer')

Pump2a=1
"""
#Correlation time!
#and plots :-)

#just Pump 2a data
Pump2a = pd.merge(MSL_data['Ace kW Pump 2a'],MSL_data['Avg. kW Pump 2a'],left_index=True, right_index=True, how='outer')
Pump2a=Pump2a.dropna() #drop nans from this set

P2amodel = LinearRegression().fit(np.array(Pump2a['Ace kW Pump 2a']).reshape((-1,1)), np.array(Pump2a['Avg. kW Pump 2a']).reshape((-1,1)))
x=np.array([min(Pump2a['Ace kW Pump 2a']), max(Pump2a['Ace kW Pump 2a'])])
y=np.array(x*P2amodel.coef_+P2amodel.intercept_)
y=[yf for ys in y for yf in ys] #For some reason you have to 'flatten' this - just do it.

plt.plot(Pump2a['Ace kW Pump 2a'], Pump2a['Avg. kW Pump 2a'])
plt.plot(x,y, linestyle='solid',color="black",)
plt.xlabel('BAS kW estimate')
plt.ylabel('Dent kW')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump2aCorrelation.png')
plt.close()

#just Pump 2b data
Pump2b = pd.merge(MSL_data['Ace kW Pump 2b'],MSL_data['Avg. kW Pump 2b'],left_index=True, right_index=True, how='outer')
Pump2b=Pump2b.dropna() #drop nans from this set

P2bmodel = LinearRegression().fit(np.array(Pump2b['Ace kW Pump 2b']).reshape((-1,1)), np.array(Pump2b['Avg. kW Pump 2b']).reshape((-1,1)))
x=np.array([min(Pump2b['Ace kW Pump 2b']), max(Pump2b['Ace kW Pump 2b'])])
y=np.array(x*P2bmodel.coef_+P2bmodel.intercept_)
y=[yf for ys in y for yf in ys] #For some reason you have to 'flatten' this - just do it.

plt.plot(Pump2b['Ace kW Pump 2b'], Pump2b['Avg. kW Pump 2b'])
plt.plot(x,y, linestyle='solid',color="black",)
plt.xlabel('BAS kW estimate')
plt.ylabel('Dent kW')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump2bCorrelation.png')
plt.close()

#just Pump 4a data
Pump4a = pd.merge(MSL_data['Ace kW Pump 4a'],MSL_data['Avg. kW Pump 4a'],left_index=True, right_index=True, how='outer')
Pump4a=Pump4a.dropna() #drop nans from this set

P4amodel = LinearRegression().fit(np.array(Pump4a['Ace kW Pump 4a']).reshape((-1,1)), np.array(Pump4a['Avg. kW Pump 4a']).reshape((-1,1)))
x=np.array([min(Pump4a['Ace kW Pump 4a']), max(Pump4a['Ace kW Pump 4a'])])
y=np.array(x*P4amodel.coef_+P4amodel.intercept_)
y=[yf for ys in y for yf in ys] #For some reason you have to 'flatten' this - just do it.

plt.plot(Pump4a['Ace kW Pump 4a'], Pump4a['Avg. kW Pump 4a'])
plt.plot(x,y, linestyle='solid',color="black",)
plt.xlabel('BAS kW estimate')
plt.ylabel('Dent kW for Pump 4a')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump4aCorrelation.png')
plt.close()

#just Pump 4b data
Pump4b = pd.merge(MSL_data['Ace kW Pump 4b'],MSL_data['Avg. kW Pump 4b'],left_index=True, right_index=True, how='outer')
Pump4b=Pump4b.dropna() #drop nans from this set

P4bmodel = LinearRegression()
P4bmodel = LinearRegression().fit(np.array(Pump4b['Ace kW Pump 4b']).reshape((-1,1)), np.array(Pump4b['Avg. kW Pump 4b']).reshape((-1,1)))
x=np.array([min(Pump4b['Ace kW Pump 4b']), max(Pump4b['Ace kW Pump 4b'])])
y=np.array(x*P4bmodel.coef_+P4bmodel.intercept_)
y=[yf for ys in y for yf in ys] #For some reason you have to 'flatten' this - just do it.

#If you need the R-squared here is the code for that:
r2=P4amodel.score(np.array(Pump4a['Ace kW Pump 4a']).reshape((-1,1)), np.array(Pump4a['Avg. kW Pump 4a']).reshape((-1,1)))
#print(r2)

plt.plot(Pump4b['Ace kW Pump 4b'], Pump4b['Avg. kW Pump 4b'])
plt.plot(x,y, linestyle='solid',color="black",markersize=0.5)
plt.xlabel('BAS kW estimate')
plt.ylabel('Dent kW for Pump 4b')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump4bCorrelation.png')
plt.close()

# #just Pump 1a data
# Pump1a = pd.merge(MSL_data['Ace kW Pump 1a'],MSL_data['Avg. kW Pump 1a'],left_index=True, right_index=True, how='outer')
# Pump1a=Pump1a.dropna() #drop nans from this set
#
# P1amodel = LinearRegression()
# P1amodel = LinearRegression().fit(np.array(Pump1a['Ace kW Pump 1a']).reshape((-1,1)), np.array(Pump1a['Avg. kW Pump 1a']).reshape((-1,1)))
# x=np.array([min(Pump1a['Ace kW Pump 1a']), max(Pump1a['Ace kW Pump 1a'])])
# y=np.array(x*P1amodel.coef_+P1amodel.intercept_)
# y=[yf for ys in y for yf in ys] #For some reason you have to 'flatten' this - just do it.
#
# plt.plot(Pump1a['Ace kW Pump 1a'], Pump1a['Avg. kW Pump 1a'])
# plt.plot(x,y, linestyle='solid',color="black",)
# plt.xlabel('BAS kW estimate')
# plt.ylabel('Dent kW for Pump 1')
# plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump1Correlation.png')
# plt.close()

#just AHU19 data
AHU19 = pd.merge(MSL_data['BAS kW AHU19'],MSL_data['Avg. kW AHU19'],left_index=True, right_index=True, how='outer')
AHU19=AHU19.dropna() #drop nans from this set

AHU19model = LinearRegression()
AHU19model = LinearRegression().fit(np.array(AHU19['BAS kW AHU19']).reshape((-1,1)), np.array(AHU19['Avg. kW AHU19']).reshape((-1,1)))
x=np.array([min(AHU19['BAS kW AHU19']), max(AHU19['BAS kW AHU19'])])
y=np.array(x*AHU19model.coef_+AHU19model.intercept_)
y=[yf for ys in y for yf in ys] #For some reason you have to 'flatten' this - just do it.

plt.plot(AHU19['BAS kW AHU19'], AHU19['Avg. kW AHU19'])
plt.plot(x,y, linestyle='solid',color="black",)
plt.xlabel('BAS AHU19 kW estimate')
plt.ylabel('Dent Power data for AHU 19')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\AHU19Correlation.png')
plt.close()

#just CT1 data
CT1 = pd.merge(MSL_data['BAS kW CT1'],MSL_data['Avg. kW CT1'],left_index=True, right_index=True, how='outer')
CT1=CT1.dropna() #drop nans from this set

CT1model = LinearRegression()
CT1model = LinearRegression(fit_intercept=False).fit(np.array(CT1['BAS kW CT1']).reshape((-1,1)), np.array(CT1['Avg. kW CT1']).reshape((-1,1)))
x=np.array([min(CT1['BAS kW CT1']), max(CT1['BAS kW CT1'])])
y=np.array(x*CT1model.coef_+CT1model.intercept_)
y=[yf for ys in y for yf in ys] #For some reason you have to 'flatten' this - just do it.

plt.plot(CT1['BAS kW CT1'], CT1['Avg. kW CT1'])
plt.plot(x,y, linestyle='solid',color="black",)
plt.xlabel('BAS Cooling Tower 1')
plt.ylabel('Dent Power data for Cooling Tower 1')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\CT1Correlation.png')
plt.close()
#
#just CT2 data
CT2 = pd.merge(MSL_data['BAS kW CT2'],MSL_data['Avg. kW CT2'],left_index=True, right_index=True, how='outer')
CT2=CT2.dropna() #drop nans from this set

CT2model = LinearRegression()
CT2model = LinearRegression(fit_intercept=False).fit(np.array(CT2['BAS kW CT2']).reshape((-1,1)), np.array(CT2['Avg. kW CT2']).reshape((-1,1)))
x=np.array([min(CT2['BAS kW CT2']), max(CT2['BAS kW CT2'])])
y=np.array(x*CT2model.coef_+CT2model.intercept_)
y=[yf for ys in y for yf in ys] #For some reason you have to 'flatten' this - just do it.

plt.plot(CT2['BAS kW CT2'], CT2['Avg. kW CT2'])
plt.plot(x,y, linestyle='solid',color="black",)
plt.xlabel('BAS Cooling Tower 2')
plt.ylabel('Dent Power data for Cooling Tower 2')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\CT2Correlation.png')
plt.close()

#just HRU data
HRU = pd.merge(MSL_data['BAS kW HRU'],MSL_data['Avg. kW HRU'],left_index=True, right_index=True, how='outer')
HRU=HRU.dropna() #drop nans from this set

HRUmodel = LinearRegression()
HRUmodel = LinearRegression().fit(np.array(HRU['BAS kW HRU']).reshape((-1,1)), np.array(HRU['Avg. kW HRU']).reshape((-1,1)))
x=np.array([min(HRU['BAS kW HRU']), max(HRU['BAS kW HRU'])])
y=np.array(x*HRUmodel.coef_+HRUmodel.intercept_)
y=[yf for ys in y for yf in ys] #For some reason you have to 'flatten' this - just do it.
#
plt.plot(HRU['BAS kW HRU'], HRU['Avg. kW HRU'])
plt.plot(x,y, linestyle='solid',color="red",)
plt.xlabel('BAS HRU kW estimate')
plt.ylabel('Dent Power data for HRU (kW)')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\HRUCorrelation.png')
plt.close()

plt.plot(MSL_data.index,MSL_data['Ace kW Pump 2a'])
plt.plot(MSL_data.index,MSL_data['Avg. kW Pump 2a'])
plt.ylabel('Power (kW)')
plt.legend(['Ace Data kW estimate','Dent Data (kW)'])
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump2aTimeserries.png')
plt.close()

plt.plot(MSL_data.index,MSL_data['Ace kW Pump 2b'])
#plt.plot(MSL_data.index,MSL_data['Ace kW Pump 2a'])
plt.plot(MSL_data.index,MSL_data['Avg. kW Pump 2b'])
plt.ylabel('Power (kW)')
#plt.legend(['Pump 2b','Pump 2b'])
plt.legend(['Ace Data kW estimate','Dent Data (kW)'])
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump2bTimeserries.png')
plt.close()

plt.plot(MSL_data.index,MSL_data['Ace kW Pump 4a'])
plt.plot(MSL_data.index,MSL_data['Avg. kW Pump 4a'])
plt.ylabel('Power (kW)')
plt.legend(['Ace Data kW estimate','Dent Data (kW)'])
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump4aTimeserries.png')
plt.close()

plt.plot(MSL_data.index,MSL_data['Ace kW Pump 4b'])
plt.plot(MSL_data.index,MSL_data['Avg. kW Pump 4b'])
plt.ylabel('Power (kW)')
plt.legend(['BAS kW estimate','Dent Data (kW)'])
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump4bTimeserries.png')
plt.close()

# plt.plot(MSL_data.index,MSL_data['Ace kW Pump 1a'])
# plt.plot(MSL_data.index,MSL_data['Avg. kW Pump 1a'])
# plt.legend(['Ace Data (Pump 1a feedback)','Dent Data (kW)'])
# plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump1aTimeserries.png')
# plt.close()

plt.plot(MSL_data.index,MSL_data['BAS kW AHU19'])
plt.plot(MSL_data.index,MSL_data['Avg. kW AHU19'])
plt.ylabel('Power (kW)')
plt.legend(['Ace Data (AHU19 Supply Fan)','Dent Data (all AHU19)'])
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\AHU19Timeserries.png')
plt.close()

plt.plot(MSL_data.index,MSL_data['BAS kW HRU'])
plt.plot(MSL_data.index,MSL_data['Avg. kW HRU'])
plt.legend(['Ace Data','Dent Data'])
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\HRUTimeserries.png')
plt.close()

plt.plot(MSL_data.index,MSL_data['BAS kW CT1'])
plt.plot(MSL_data.index,MSL_data['Avg. kW CT1'])
plt.legend(['Ace Data','Dent Data'])
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\CT1Timeserries.png')
plt.close()

plt.plot(MSL_data.index,MSL_data['BAS kW CT2'])
plt.plot(MSL_data.index,MSL_data['Avg. kW CT2'])
plt.legend(['Ace Data','Dent Data'])
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\CT2Timeserries.png')
plt.close()

#Export Linear Regressions!
#Format: Equipment name, slope, intercept, rsquared

# #Save in a .csv
# with open(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\RegressionParameters.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Equipment name', 'slope', 'intercept', 'rsquared'])
#     writer.writerow(['Pump4a',float(P4amodel.coef_),float(P4amodel.intercept_),P4amodel.score(np.array(Pump4a['Ace kW Pump 4a']).reshape((-1,1)),
#                                                                                  np.array(Pump4a['Avg. kW Pump 4a']).reshape((-1,1)))])
#     writer.writerow(['Pump4b', float(P4bmodel.coef_),float(P4bmodel.intercept_),P4bmodel.score(np.array(Pump4b['Ace kW Pump 4b']).reshape((-1,1)),
#                                                                                 np.array(Pump4b['Avg. kW Pump 4b']).reshape((-1,1)))])
#     writer.writerow(['Pump2a', float(P2amodel.coef_), float(P2amodel.intercept_), P2amodel.score(np.array(Pump2a['Ace kW Pump 2a']).reshape((-1, 1)),
#                                                               np.array(Pump2a['Avg. kW Pump 2a']).reshape((-1, 1)))])
#     writer.writerow(['Pump2b', float(P2bmodel.coef_), float(P2bmodel.intercept_), P2bmodel.score(np.array(Pump2b['Ace kW Pump 2b']).reshape((-1, 1)),
#                                                               np.array(Pump2b['Avg. kW Pump 2b']).reshape((-1, 1)))])
#     writer.writerow(['HRU', float(HRUmodel.coef_), float(HRUmodel.intercept_), HRUmodel.score(np.array(HRU['BAS kW HRU']).reshape((-1, 1)),
#                                                              np.array(HRU['Avg. kW HRU']).reshape((-1, 1)))])
#     writer.writerow(['AHU19', float(AHU19model.coef_), float(AHU19model.intercept_), AHU19model.score(np.array(AHU19['BAS kW AHU19']).reshape((-1, 1)),
#                                                               np.array(AHU19['Avg. kW AHU19']).reshape((-1, 1)))])
#     writer.writerow(['CoolingTower1', float(CT1model.coef_), float(CT1model.intercept_), CT1model.score(np.array(CT1['BAS kW CT1']).reshape((-1, 1)),
#                                                               np.array(CT1['Avg. kW CT1']).reshape((-1, 1)))])
#     writer.writerow(['CoolingTower2', float(CT2model.coef_), float(CT2model.intercept_), CT2model.score(np.array(CT2['BAS kW CT2']).reshape((-1, 1)),
#                                                              np.array(CT2['Avg. kW CT2']).reshape((-1, 1)))])
"""