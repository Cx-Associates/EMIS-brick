"""
Get data from Ace and dents
Combine data
Calculate estimated kW using proxy formulas
Create some correlation plots and correlation values (and save)
Create some time series plots (and save)
"""

import os
import requests
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


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


#some places and specifics
dentdatapath=[r"F:/PROJECTS/1715 Main Street Landing EMIS Pilot/data/Dent data pull, 2024-01-05/raw data/E876C-01.csv",
              r"F:/PROJECTS/1715 Main Street Landing EMIS Pilot/data/Dent data pull, 2024-01-05/raw data/E876D-01.csv",
              r"F:/PROJECTS/1715 Main Street Landing EMIS Pilot/data/Dent data pull, 2024-01-05/raw data/E876F-01.csv"]
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

    # create 'time' column
    MSL_data1['CombinedDatetime']=pd.to_datetime(MSL_data1['Date '] + ' ' + MSL_data1['End Time '])
    #^for some reason that has us off by 4 hours?
    MSL_data1.set_index('CombinedDatetime', inplace=True)
    MSL_data1.index = MSL_data1.index.tz_localize('US/Eastern', ambiguous='NaT')
    # Handle ambiguous time error by dropping rows with NaT values in the index
    MSL_data1 = MSL_data1[MSL_data1.index.notnull()]

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

#Pump/fan nameplates
Nameplate= {'Equipt':['Pump1a', 'Pump1b', 'Pump2a', 'Pump2b', 'Pump4a', 'Pump4b', 'HRUSupplyFan', 'HRUReturnFan',
                        'AHU19SupplyFan', 'AHU19ReturnFan'], 'hp':[20, 15, 25, 25, 7.5, 7.5, 10, 10, 7.5, 10]}
nameplate=pd.DataFrame(Nameplate)

start = "2023-11-10"
end = "2024-03-31"

#Ace Data locations
#Todo: make this a file you pull from instead of hard coded.
str = [fr'/cxa_main_st_landing/2404:9-240409/analogOutput/5/timeseries?start_time={start}&end_time={end}', #Pump 4a VFD Output
fr'/cxa_main_st_landing/2404:9-240409/analogOutput/6/timeseries?start_time={start}&end_time={end}', #Pump 4b VFD Output
fr'/cxa_main_st_landing/2404:9-240409/analogInput/16/timeseries?start_time={start}&end_time={end}', #Primary Hot Water Supply Temp_2
fr'/cxa_main_st_landing/2404:9-240409/analogInput/15/timeseries?start_time={start}&end_time={end}', #Primary Hot Water Return Temp_2
fr'/cxa_main_st_landing/2404:9-240409/analogOutput/3/timeseries?start_time={start}&end_time={end}', #Boiler 1% signal
fr'/cxa_main_st_landing/2404:9-240409/analogOutput/4/timeseries?start_time={start}&end_time={end}', #Boiler 2% signal
fr'/cxa_main_st_landing/2404:9-240409/binaryOutput/6/timeseries?start_time={start}&end_time={end}', #Boiler 1 Status
fr'/cxa_main_st_landing/2404:9-240409/binaryOutput/7/timeseries?start_time={start}&end_time={end}', #Boiler 2 Status'
fr'/cxa_main_st_landing/2404:7-240407/analogValue/11/timeseries?start_time={start}&end_time={end}', #chilled water power meter
fr'/cxa_main_st_landing/2404:7-240407/analogOutput/4/timeseries?start_time={start}&end_time={end}', #pump 2a-b VFD output
fr'/cxa_main_st_landing/2404:7-240407/binaryOutput/12/timeseries?start_time={start}&end_time={end}', #pump 2a activity (binary)
fr'/cxa_main_st_landing/2404:7-240407/binaryOutput/13/timeseries?start_time={start}&end_time={end}', #pump 2b activity (binary)
fr'/cxa_main_st_landing/2404:2-240402/analogInput/10/timeseries?start_time={start}&end_time={end}', #P1a Feedback
fr'/cxa_main_st_landing/2404:2-240402/analogInput/11/timeseries?start_time={start}&end_time={end}', #P1b Feedback
fr'/cxa_main_st_landing/2404:2-240402/analogOutput/3/timeseries?start_time={start}&end_time={end}', #P1 VFD Signal
fr'/cxa_main_st_landing/2404:7-240407/binaryInput/9/timeseries?start_time={start}&end_time={end}', #Pump 3a status (binary)
fr'/cxa_main_st_landing/2404:7-240407/binaryInput/10/timeseries?start_time={start}&end_time={end}', #Pump 3b status (binary)
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
fr'/cxa_main_st_landing/2404:10-240410/analogOutput/8/timeseries?start_time={start}&end_time={end}', #HRU Supplyfan VFD output
fr'/cxa_main_st_landing/2404:10-240410/analogOutput/2/timeseries?start_time={start}&end_time={end}', #HRU Exhaust Fan VFD speed
fr'/cxa_main_st_landing/2404:10-240410/binaryInput/1/timeseries?start_time={start}&end_time={end}', #HRU Exhaust Fan Status
fr'/cxa_main_st_landing/2404:10-240410/binaryInput/9/timeseries?start_time={start}&end_time={end}', #HRU Supply Fan Status
fr'/cxa_main_st_landing/2404:3-240403/analogOutput/3/timeseries?start_time={start}&end_time={end}', #AHU19 Supply fan VFD
fr'/cxa_main_st_landing/2404:3-240403/analogOutput/5/timeseries?start_time={start}&end_time={end}', #Exhaust fan 1 VFD speed
fr'/cxa_main_st_landing/2404:3-240403/analogOutput/6/timeseries?start_time={start}&end_time={end}', #Exhaust fan 2 VFD speed
fr'/cxa_main_st_landing/2404:3-240403/analogOutput/2/timeseries?start_time={start}&end_time={end}', #Heat Recovery Wheel VFD
fr'/cxa_main_st_landing/2404:3-240403/binaryInput/6/timeseries?start_time={start}&end_time={end}', #Heat Recovery Wheel Status
fr'/cxa_main_st_landing/2404:3-240403/analogValue/9/timeseries?start_time={start}&end_time={end}', #Exhaust fan CFM
fr'/cxa_main_st_landing/2404:3-240403/analogValue/16/timeseries?start_time={start}&end_time={end}', #Total Cool Request from Zones
fr'/cxa_main_st_landing/2404:3-240403/analogValue/17/timeseries?start_time={start}&end_time={end}', #Total Heat Request from Zones
fr'/cxa_main_st_landing/2404:7-240407/binaryInput/18/timeseries?start_time={start}&end_time={end}', #Pump 2a status
fr'/cxa_main_st_landing/2404:7-240407/binaryInput/19/timeseries?start_time={start}&end_time={end}',#Pump 2b status
fr'/cxa_main_st_landing/2404:9-240409/binaryOutput/12/timeseries?start_time={start}&end_time={end}',#Pump 4a s/s
fr'/cxa_main_st_landing/2404:9-240409/binaryOutput/13/timeseries?start_time={start}&end_time={end}']#Pump 4b s/s

#Ace Data descriptions #Todo: Add statuses when available
headers = ['Pump 4a VFD Output',
         'Pump 4b VFD Output',
         'Primary Hot Water Supply Temp_2',
         'Primary Hot Water Return Temp_2',
         'Boiler 1% signal',
         'Boiler 2% signal',
         'Boiler 1 status',
         'Boiler 2 status',
         'Chilled water power meter',
         'Pump 2a-b VFD output',
         'Pump 2a activity',
         'Pump 2b activity',
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
         'AHU19 Exhaust fan 1 VFD speed',
         'AHU19 Exhaust fan 2 VFD speed',
         'AHU19 Heat Recovery Wheel VFD',
         'AHU19 Heat Recovery Wheel Status',
         'AHU19 Exhaust fan CFM',
         'Total Cool Request from Zones',
         'Total Heat Request from Zones',
         'Pump 2a status',
         'Pump 2b status',
         'Pump 4a s/s',
         'Pump 4b s/s']

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
Ace_data['pump2a'] = Ace_data['Pump 2a activity']*Ace_data['Pump 2a-b VFD output']*Ace_data['Pump 2a status'] #this should be the BAS pump 2a VFD Output
Ace_data['pump2b'] = Ace_data['Pump 2b activity']*Ace_data['Pump 2a-b VFD output']*Ace_data['Pump 2b status'] #same for pump 2b

#Calculate kW from dent data
#Assume a PF of 0.8 for now:
PF=0.8
MSL_data['Avg. kW Pump 4a'] = MSL_data['Avg. Volt Pump 4a']*MSL_data['Avg. Amp Pump 4a']*3**.5*PF/1000
MSL_data['Avg. kW Pump 4b'] = MSL_data['Avg. Volt Pump 4b']*MSL_data['Avg. Amp Pump 4b']*3**.5*PF/1000
MSL_data['Avg. kW AHU19'] = MSL_data['Avg. Volt AHU19']*MSL_data['Avg. Amp AHU19']*3**.5*PF/1000
MSL_data['Avg. kW Pump 1a'] = MSL_data['Avg. Volt Pump 1a']*MSL_data['Avg. Amp Pump 1a']*3**.5*PF/1000
MSL_data['Avg. kW Pump 2b'] = MSL_data['Avg. Volt Pump 2b']*MSL_data['Avg. Amp Pump 2b']*3**.5*PF/1000
MSL_data['Avg. kW Pump 2a'] = MSL_data['Avg. Volt Pump 2a']*MSL_data['Avg. Amp Pump 2a']*3**.5*PF/1000
MSL_data['Avg. kW L1 HRU'] = MSL_data['Avg. VoltL1 HRU']*MSL_data['Avg. AmpL1 HRU']*3**.5*PF/1000
MSL_data['Avg. kW L2 HRU'] = MSL_data['Avg. VoltL2 HRU']*MSL_data['Avg. AmpL2 HRU']*3**.5*PF/1000
MSL_data['Avg. kW AHU9'] = MSL_data['Avg. Volt AHU9']*MSL_data['Avg. Amp AHU9']*3**.5*PF/1000

#Calculate expected power from AceData
#=pump nameplate HP *0.745699872*%pump speed^2.5
Ace_data['Ace kW Pump 4a']=get_hp('Pump4a',Nameplate)*0.745699872*(Ace_data['Pump 4a VFD Output']/100)**2.5*Ace_data['Pump 4a s/s']
Ace_data['Ace kW Pump 4b']=get_hp('Pump4b',Nameplate)*0.745699872*(Ace_data['Pump 4b VFD Output']/100)**2.5*Ace_data['Pump 4b s/s']
Ace_data['Ace kW Pump 1a']=get_hp('Pump1a',Nameplate)*0.745699872*(Ace_data['Pump 1a feedback']/100)**2.5
Ace_data['Ace kW Pump 2b']=get_hp('Pump2b',Nameplate)*0.745699872*(Ace_data['pump2b']/100)**2.5
Ace_data['Ace kW Pump 2a']=(get_hp('Pump2a',Nameplate)*0.745699872*(Ace_data['pump2a']/100)**2.5)

#15 minute Ace data averages
Ace_15min = Ace_data[head].resample('15T').mean()

#comine ace and Dent (MSL) data
MSL_data = pd.merge(MSL_data, Ace_data, left_index=True, right_index=True, how='outer')

#this is risky - drop rows with NaNs
MSL_data = MSL_data.dropna()

#Correlation time!
#and plots :-)
P2amodel = LinearRegression()
P2amodel = LinearRegression().fit(np.array(MSL_data['Ace kW Pump 2a']).reshape((-1,1)), np.array(MSL_data['Avg. kW Pump 2a']).reshape((-1,1)))
x=np.array([min(MSL_data['Ace kW Pump 2a']), max(MSL_data['Ace kW Pump 2a'])])
y=np.array(x*P2amodel.coef_+P2amodel.intercept_)
y=[yf for ys in y for yf in ys] #For some reason you have to 'flatten' this - just do it.

plt.plot(MSL_data['Ace kW Pump 2a'], MSL_data['Avg. kW Pump 2a'])
plt.plot(x,y, linestyle='solid',color="black",)
plt.xlabel('BAS kW estimate')
plt.ylabel('Dent kW')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump2aCorrelation.png')
plt.close()

P2bmodel = LinearRegression()
P2bmodel = LinearRegression().fit(np.array(MSL_data['Ace kW Pump 2b']).reshape((-1,1)), np.array(MSL_data['Avg. kW Pump 2b']).reshape((-1,1)))
x=np.array([min(MSL_data['Ace kW Pump 2b']), max(MSL_data['Ace kW Pump 2b'])])
y=np.array(x*P2bmodel.coef_+P2bmodel.intercept_)
y=[yf for ys in y for yf in ys] #For some reason you have to 'flatten' this - just do it.

plt.plot(MSL_data['Ace kW Pump 2b'], MSL_data['Avg. kW Pump 2b'])
plt.plot(x,y, linestyle='solid',color="black",)
plt.xlabel('BAS kW estimate')
plt.ylabel('Dent kW')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump2bCorrelation.png')
plt.close()

P4amodel = LinearRegression()
P4amodel = LinearRegression().fit(np.array(MSL_data['Ace kW Pump 4a']).reshape((-1,1)), np.array(MSL_data['Avg. kW Pump 4a']).reshape((-1,1)))
x=np.array([min(MSL_data['Ace kW Pump 4a']), max(MSL_data['Ace kW Pump 4a'])])
y=np.array(x*P4amodel.coef_+P4amodel.intercept_)
y=[yf for ys in y for yf in ys] #For some reason you have to 'flatten' this - just do it.

plt.plot(MSL_data['Ace kW Pump 4a'], MSL_data['Avg. kW Pump 4a'])
plt.plot(x,y, linestyle='solid',color="black",)
plt.xlabel('BAS kW estimate')
plt.ylabel('Dent kW for Pump 4a')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump4aCorrelation.png')
plt.close()

P4bmodel = LinearRegression()
P4bmodel = LinearRegression().fit(np.array(MSL_data['Ace kW Pump 4b']).reshape((-1,1)), np.array(MSL_data['Avg. kW Pump 4b']).reshape((-1,1)))
x=np.array([min(MSL_data['Ace kW Pump 4b']), max(MSL_data['Ace kW Pump 4b'])])
y=np.array(x*P4bmodel.coef_+P4bmodel.intercept_)
y=[yf for ys in y for yf in ys] #For some reason you have to 'flatten' this - just do it.

plt.plot(MSL_data['Ace kW Pump 4b'], MSL_data['Avg. kW Pump 4b'])
plt.plot(x,y, linestyle='solid',color="black",markersize=0.5)
plt.xlabel('BAS kW estimate')
plt.ylabel('Dent kW for Pump 4b')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump4bCorrelation.png')
plt.close()

P1amodel = LinearRegression()
P1amodel = LinearRegression().fit(np.array(MSL_data['Ace kW Pump 1a']).reshape((-1,1)), np.array(MSL_data['Avg. kW Pump 1a']).reshape((-1,1)))
x=np.array([min(MSL_data['Ace kW Pump 1a']), max(MSL_data['Ace kW Pump 1a'])])
y=np.array(x*P1amodel.coef_+P1amodel.intercept_)
y=[yf for ys in y for yf in ys] #For some reason you have to 'flatten' this - just do it.

plt.plot(MSL_data['Ace kW Pump 1a'], MSL_data['Avg. kW Pump 1a'])
plt.plot(x,y, linestyle='solid',color="black",)
plt.xlabel('BAS kW estimate')
plt.ylabel('Dent kW for Pump 1')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump1Correlation.png')
plt.close()

plt.plot(MSL_data['AHU19 supply fan VFD output'], MSL_data['Avg. Amp AHU19'])
plt.xlabel('BAS AHU19 Supply Fan')
plt.ylabel('Dent Power data for AHU 19')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\AHU19Correlation.png')
plt.close()

HRUmodel = LinearRegression()
HRUmodel = LinearRegression().fit(np.array(MSL_data['HRU supply fan']).reshape((-1,1)), np.array(MSL_data['Avg. kW L1 HRU']).reshape((-1,1)))
x=np.array([min(MSL_data['HRU supply fan']), max(MSL_data['HRU supply fan'])])
y=np.array(x*HRUmodel.coef_+HRUmodel.intercept_)
y=[yf for ys in y for yf in ys] #For some reason you have to 'flatten' this - just do it.

plt.plot(MSL_data['HRU supply fan'], MSL_data['Avg. kW L1 HRU'])
plt.plot(MSL_data['HRU supply fan'], MSL_data['Avg. kW L2 HRU'])
plt.plot(x,y, linestyle='solid',color="red",)
plt.xlabel('HRU Supply fan')
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
plt.legend(['Ace Data kW estimate','Dent Data (kW)'])
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump4bTimeserries.png')
plt.close()

plt.plot(MSL_data.index,MSL_data['Ace kW Pump 1a'])
plt.plot(MSL_data.index,MSL_data['Avg. kW Pump 1a'])
plt.legend(['Ace Data (Pump 1a feedback)','Dent Data (kW)'])
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Pump1aTimeserries.png')
plt.close()

plt.plot(MSL_data.index,MSL_data['kW AHU19'])
plt.plot(MSL_data.index,MSL_data['Avg. kW AHU19'])
plt.ylabel('Power (kW)')
plt.legend(['Ace Data (AHU19 Supply Fan)','Dent Data (all AHU19)'])
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\AHU19Timeserries.png')
plt.close()

plt.plot(MSL_data.index,MSL_data['HRU supply fan'])
plt.plot(MSL_data.index,MSL_data['Avg. AmpL1 HRU'])
plt.plot(MSL_data.index,MSL_data['Avg. AmpL2 HRU'])
plt.legend(['Ace Data (HRU Supply Fan)','Dent Data Phase 1','Dent Data Phase 2'])
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\HRUTimeserries.png')
plt.close()
