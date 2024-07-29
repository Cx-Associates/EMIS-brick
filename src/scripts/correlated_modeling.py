
#Copied from new_model - sorting through it
import os
import requests
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.linear_model import LinearRegression

#from src.utils import Project
#from config_MSL import config_dict

from Dent_compile import P2amodel, P2bmodel, P4amodel, P4bmodel, P1amodel, HRUmodel, MSL_data, get_hp

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

env_filename = 'api_keys.yml'
f_drive_path = 'F:/PROJECTS/1715 Main Street Landing EMIS Pilot/code/API keys'
env_filepath = os.path.join(f_drive_path, env_filename)
timezone='US/Eastern'
start = "2023-11-10"
end = "2024-03-31"
ACE_data = pd.DataFrame()

#Ace Data locations #Todo: Add statuses when available
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
fr'/cxa_main_st_landing/2404:9-240409/binaryInput/19/timeseries?start_time={start}&end_time={end}', #P4B Status
fr'/cxa_main_st_landing/2404:9-240409/binaryInput/18/timeseries?start_time={start}&end_time={end}', #P4A Status
fr'/cxa_main_st_landing/2404:3-240403/binaryInput/3/timeseries?start_time={start}&end_time={end}'] #AHU19 Supply Fan Status

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
         'Pump 4b Status',
         'Pump 4a Status'
         'AHU19 Supply Fan Status']

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
            #raise Exception(msg)

ACE_data.to_csv('ACE_Data_5.csv') #Uncomment this out when the start and end dates have changed or any change in data is expected. This will write over the existing file.

#Pump/fan nameplates
Nameplate= {'Equipt':['Pump1a', 'Pump1b', 'Pump2a', 'Pump2b', 'Pump4a', 'Pump4b', 'HRUSupplyFan', 'HRUReturnFan',
                        'AHU19SupplyFan', 'AHU19ReturnFan', 'Pump3a', 'Pump3b', "CTFan1", "CTFan2", "AHU19EF1", "AHU19EF2","AHU19SF", "AHU19HRW"], 'hp':[20, 15, 25, 25, 7.5, 7.5, 10, 10, 7.5, 10, 7.5, 7.5, 15, 15, 10, 10, 7.5, 0.1]} #Todo:Add remaining equipment
nameplate=pd.DataFrame(Nameplate)

#Calculating kW from BMS information

#Heating system
ACE_data['Pump 4a kW (Formula Based)'] = get_hp('Pump4a',Nameplate)*0.745699872*(ACE_data['Pump 4a VFD Output']/100)**2.5 #Todo: Add status when available
ACE_data['Pump 4b kW (Formula Based)'] = get_hp('Pump4b',Nameplate)*0.745699872*(ACE_data['Pump 4b VFD Output']/100)**2.5 #Todo: Add status when available

#Chilled water system
ACE_data['Pump 1a kW (Formula Based)'] = get_hp('Pump1a',Nameplate)*0.745699872*(MSL_data['Pump 1a feedback']/100)**2.5 #Todo: Add status when available
ACE_data['Pump 1b kW (Formula Based)'] = get_hp('Pump1b', Nameplate)*0.745699872*(MSL_data['Pump 1b feedback']/100)**2.5 #Todo: Add status when available
ACE_data['Pump 2b kW (Formula Based)'] = ACE_data['Pump 2b activity']*get_hp('Pump2b',Nameplate)*0.745699872*(ACE_data['Pump 2a-b VFD output']/100)**2.5
ACE_data['Pump 2a kW (Formula Based)'] = ACE_data['Pump 2a activity']*(get_hp('Pump2a',Nameplate)*0.745699872*(ACE_data['Pump 2a-b VFD output']/100)**2.5)
ACE_data['Pump 3a kW (Formula Based)'] = ACE_data['Pump 3a status']*(get_hp('Pump3a', Nameplate))*0.745699872
ACE_data['Pump 3b kW (Formula Based)'] = ACE_data['Pump 3b status']*(get_hp('Pump3a', Nameplate))*0.745699872
ACE_data['Tower Fan 1 kW (Formula Based)'] = ACE_data['Cooling tower Fan 1 Status']*(get_hp('CTFan1', Nameplate))*0.745699872*(ACE_data['Cooling tower fan %speed']/100)**2.5
ACE_data['Tower Fan 2 kW (Formula Based)'] = ACE_data['Cooling tower Fan 2 Status']*(get_hp('CTFan2', Nameplate))*0.745699872*(ACE_data['Cooling tower fan %speed']/100)**2.5
ACE_data['Chiller kW'] = ACE_data['Chiller status'] * ACE_data['Chilled water power meter']

#HRU
ACE_data['HRU Supply Fan kW (Formula Based)'] = ACE_data['HRU Supply Fan Status']*(get_hp('HRUSupplyFan',Nameplate))*0.745699872*(ACE_data['HRU supply fan VFD output']/100)**2.5
ACE_data['HRU Exhaust Fan kW (Formula Based)'] = ACE_data['HRU Exhaust Fan Status']*(get_hp('HRUReturnFan',Nameplate))*0.745699872*(ACE_data['HRU Exhaust fan VFD output']/100)**2.5
ACE_data['HRU Total kW (Formula Based)'] = ACE_data['HRU Exhaust Fan kW (Formula Based)'] + ACE_data['HRU Supply Fan kW (Formula Based)']

#AHU19
#ACE_data['AHU 19 EF1 kW (Formula Based)'] = ACE_data['AHU 19 EF1 Status']*(get_hp('AHU19EF1', Nameplate))*0.745699872*(ACE_data['AHU19 Exhaust fan 1 VFD speed']/100)**2.5
#ACE_data['AHU 19 EF2 kW (Formula Based)'] = ACE_data['AHU 19 EF2 Status']*(get_hp('AHU19EF2', Nameplate))*0.745699872*(ACE_data['AHU19 Exhaust fan 2 VFD speed']/100)**2.5
#ACE_data['AHU 19 SF kW (Formula Based)'] = ACE_data['AHU 19 SF Status']*(get_hp('AHU19SF', Nameplate))*0.745699872*(ACE_data['AHU19 supply fan VFD output']/100)**2.5
#ACE_data['AHU 19 HRW kW (Formula Based)'] = ACE_data['AHU19 Heat Recovery Wheel Status']*(get_hp('AHU19HRW', Nameplate))*0.745699872*(ACE_data['AHU19 Heat Recovery Wheel VFD']/100)**2.5


#Calculating the Dent correlated kW consumption #Todo: Uncomment this out and use 15 min average data for correlation and then calculate total power consumption for each system ("ACE_data" df is currently in 5 min intervals)
#ACE_data['Pump 4a kW (Dent Correlated)'] = (ACE_data['Pump 4a kW (Formula Based)']*P4amodel.coef_) + P4amodel.intercept_
#ACE_data['Pump 4b kW (Dent Correlated)'] = (ACE_data['Pump 4b kW (Formula Based)']*P4bmodel.coef_) + P4bmodel.intercept_
#ACE_data['Pump 1a kW (Dent Correlated)'] = (ACE_data['Pump 1a kW (Formula Based)']*P1amodel.coef_) + P1amodel.intercept_
#ACE_data['Pump 2a kW (Dent Correlated)'] = (ACE_data['Pump 2a kW (Formula Based)']*P2amodel.coef_) + P2amodel.intercept_
#ACE_data['Pump 2b kW (Dent Correlated)'] = (ACE_data['Pump 2b kW (Formula Based)']*P2bmodel.coef_) + P2bmodel.intercept_
#ACE_data['HRU Exhaust Fan kW (Dent Correlated)'] = (ACE_data['HRU Total kW (Formula Based)']*HRUmodel.coef_) + HRUmodel.intercept_

#Todo: Add remaining DENT correlation when data is available

