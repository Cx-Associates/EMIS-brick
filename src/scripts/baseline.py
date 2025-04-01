"""
CODE FINDS A LINEAR RELATIONSHIP BETWEEN HEATING DEGREE DAYS AND HEATING SYSTEM ENERGY USAGE
OUTPUTS MODEL TO A .CSV FILE THAT CAN BE READ BY OTHER CODES TO COMPARE HEATING SYSTEM USAGE WITH
THE USAGE DURING THE BAESLINE PERIOD.

CODE ALSO HAS THE FRAMEWORK TO DO THIS FOR THE VENTILATION SYSTEM AND COOLING SYSTEM.
"""

#Todo: Delete whatever we don't end up using from the below imports
#Todo: For future projects have a function for baseline where the dates are adjustable
import os
import requests
import pandas as pd
import yaml
from datetime import date
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import csv
import calendar

#Define baseline period
#IF YOU UPDATE THESE DATES, ALSO UPDATE THEM IN CORRELATED_MODELING.PY!!!
timezone='US/Eastern'
start_baseline_heating = '2024-11-01'#"xx-xx-xxxx" #start of baseline period #todo: update when baseline period is determined, current dates are for heating system baseline
end_baseline_heating = '2025-02-01'#"xx-xx-xxxx" #end of baseline period #todo: update when baseline period is determined, current dates are for heating system baseline
start_baseline_vent = '2024-11-01'#"xx-xx-xxxx" #start of baseline period for ventilation #todo: update when baseline period is determined, current dates are for ventilation system baseline
end_baseline_vent = '2025-03-01'#"xx-xx-xxxx" #end of baseline period for ventilation #todo: update when baseline period is determined, current dates are for ventilation system baseline

#start_check = '' #Start check and end check should be the actual dates of baseline
#end_check = ''
start_check = pd.to_datetime(start_baseline_heating).tz_localize(timezone)
end_check = pd.to_datetime(end_baseline_vent).tz_localize(timezone) #this is the later date.  todo:make this automatically choose the last date

#Finding the balance point

#Import Correlation Parameters
corr_path = "F:/PROJECTS/1715 Main Street Landing EMIS Pilot/code/RegressionParameters.csv"
Corr_param_df = pd.DataFrame(pd.read_csv(corr_path))

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


ACE_data = pd.DataFrame() #Defining empty dataframe into which BMS data will be pulled into from ACE API

def get_value(equipment_name, data): #Todo: For next project it will be good to define al functions in the utils.py
    index = data['Equipt'].index(equipment_name)  # Find index of equipment name
    size = data['value'][index]  # Retrieve corresponding size using index
    return size

def get_hp(equipment_name, data):
    index = data['Equipt'].index(equipment_name)  # Find index of equipment name
    size = data['hp'][index]  # Retrieve corresponding size using index
    return size

#Ace Data locations
str = [fr'/cxa_main_st_landing/2404:9-240409/analogOutput/5/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Pump 4a VFD Output
fr'/cxa_main_st_landing/2404:9-240409/analogOutput/6/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Pump 4b VFD Output
fr'/cxa_main_st_landing/2404:9-240409/binaryOutput/12/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Pump 4a s/s
fr'/cxa_main_st_landing/2404:9-240409/binaryOutput/13/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Pump 4b s/s
fr'/cxa_main_st_landing/2404:9-240409/analogInput/16/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Primary Hot Water Supply Temp_2
fr'/cxa_main_st_landing/2404:9-240409/analogInput/15/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Primary Hot Water Return Temp_2
fr'/cxa_main_st_landing/2404:9-240409/analogOutput/3/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Boiler 1% signal
fr'/cxa_main_st_landing/2404:9-240409/analogOutput/4/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Boiler 2% signal
fr'/cxa_main_st_landing/2404:9-240409/binaryOutput/6/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Boiler 1 Status
fr'/cxa_main_st_landing/2404:9-240409/binaryOutput/7/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Boiler 2 Status'
fr'/cxa_main_st_landing/2404:7-240407/analogValue/11/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #chilled water power meter
fr'/cxa_main_st_landing/2404:7-240407/analogOutput/4/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #pump 2a-b VFD output
fr'/cxa_main_st_landing/2404:7-240407/binaryInput/18/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Pump 2a status
fr'/cxa_main_st_landing/2404:7-240407/binaryInput/19/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Pump 2b status
fr'/cxa_main_st_landing/2404:2-240402/analogInput/10/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #P1a Feedback
fr'/cxa_main_st_landing/2404:2-240402/analogInput/11/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #P1b Feedback
fr'/cxa_main_st_landing/2404:2-240402/analogOutput/3/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #P1 VFD Signal
fr'/cxa_main_st_landing/2404:7-240407/binaryInput/9/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Pump 3a status
fr'/cxa_main_st_landing/2404:7-240407/binaryInput/10/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Pump 3b status
fr'/cxa_main_st_landing/2404:7-240407/binaryOutput/5/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Chiller Status (binary)
fr'/cxa_main_st_landing/2404:7-240407/analogInput/8/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Chiller HX1 Flow (GPM) (only flow data we have for chiller)
fr'/cxa_main_st_landing/2404:7-240407/analogInput/21/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Chilled water supply temp (F)
fr'/cxa_main_st_landing/2404:7-240407/analogInput/20/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Chilled water return temp (F)
fr'/cxa_main_st_landing/2404:7-240407/analogInput/17/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Condenser Water Supply Temperature (F)
fr'/cxa_main_st_landing/2404:7-240407/analogInput/13/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Condenser Water Return Temperature (F)
fr'/cxa_main_st_landing/2404:2-240402/analogInput/7/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Cooling Tower Temp In (F)
fr'/cxa_main_st_landing/2404:2-240402/analogInput/8/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Cooling Tower Temp Out (F)
fr'/cxa_main_st_landing/2404:7-240407/binaryValue/11/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Cooling Tower Free Cool Status (binary)
fr'/cxa_main_st_landing/2404:2-240402/analogOutput/4/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Cooling tower fan %speed
fr'/cxa_main_st_landing/2404:2-240402/binaryInput/10/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Cooling tower Fan1 Status
fr'/cxa_main_st_landing/2404:2-240402/binaryInput/11/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Cooling tower Fan2 Status
fr'/cxa_main_st_landing/2404:10-240410/analogOutput/8/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #HRU Supply fan VFD output
fr'/cxa_main_st_landing/2404:10-240410/analogOutput/2/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #HRU Exhaust Fan VFD speed
fr'/cxa_main_st_landing/2404:10-240410/binaryInput/1/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #HRU Exhaust Fan Status
fr'/cxa_main_st_landing/2404:10-240410/binaryInput/9/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #HRU Supply Fan Status
fr'/cxa_main_st_landing/2404:3-240403/analogOutput/3/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #AHU19 Supply fan VFD
fr'/cxa_main_st_landing/2404:3-240403/binaryInput/3/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #AHU19 Supply fan Status
fr'/cxa_main_st_landing/2404:3-240403/analogOutput/5/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Exhaust fan 1 VFD speed
fr'/cxa_main_st_landing/2404:3-240403/analogOutput/6/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Exhaust fan 2 VFD speed
fr'/cxa_main_st_landing/2404:3-240403/analogOutput/2/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Heat Recovery Wheel VFD
fr'/cxa_main_st_landing/2404:3-240403/binaryInput/6/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Heat Recovery Wheel Status
fr'/cxa_main_st_landing/2404:3-240403/analogValue/9/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Exhaust fan CFM
fr'/cxa_main_st_landing/2404:3-240403/analogValue/16/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Total Cool Request from Zones
fr'/cxa_main_st_landing/2404:3-240403/analogValue/17/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #Total Heat Request from Zones
fr'/cxa_main_st_landing/2404:9-240409/binaryInput/19/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}',  #P4B Status
fr'/cxa_main_st_landing/2404:9-240409/binaryInput/18/timeseries?start_time={start_baseline_heating}&end_time={end_baseline_heating}'] #P4A Status


#Ace Data descriptions
headers = ['Pump 4a VFD Output',
         'Pump 4b VFD Output',
         'Pump 4a s/s',
         'Pump 4b s/s',
         'Primary Hot Water Supply Temp_2',
         'Primary Hot Water Return Temp_2',
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

            ACE_data = pd.DataFrame.merge(ACE_data, df, left_index=True, right_index=True, how='outer')
        else:
            msg = f'API request from ACE was unsuccessful. \n {res.reason} \n {res.content}'
            #raise Exception(msg) #Uncomment this to troubleshoot any points that are not being downloaded

#ACE_data.to_csv('ACE_Data_5.csv') #Uncomment this out for troubleshooting

##Calculating system level energy consumption for baseline period
#Pump/fan nameplates
Nameplate= {'Equipt':['Pump1a', 'Pump1b', 'Pump2a', 'Pump2b', 'Pump4a', 'Pump4b', 'HRUSupplyFan', 'HRUReturnFan',
                        'AHU19SupplyFan', 'AHU19ReturnFan', 'Pump3a', 'Pump3b', "CTFan1", "CTFan2", "AHU19EF1", "AHU19EF2","AHU19SF", "AHU19HRW"], 'hp':[20, 15, 25, 25, 7.5, 7.5, 10, 10, 7.5, 10, 7.5, 7.5, 15, 15, 10, 10, 7.5, 0.1]}
nameplate=pd.DataFrame(Nameplate)

#Boiler Nameplate
Boiler_Nameplate = {'Equipt':['Boiler1_capacity', 'Boiler2_capacity', 'Boiler1_Eff', 'Boiler2_Eff'], 'value':[2047, 2081, 0.878, 0.891]}

Baseline_heating = pd.DataFrame() #Dataframe which will store all calculated energy consumption and any data needed for reporting
Baseline_vent = pd.DataFrame() #Dataframe which will store all calculated energy consumption and any data needed for reporting
Baseline_cooling = pd.DataFrame() #Dataframe which will store all calculated energy consumption and any data needed for reporting

#Todo: For future projects, will be good if we do system level correlations instead of doing equipment level, will reduce steps and problems with loss of data
#Todo: For future projects, we should have different py files for each system so that if data is missing for certain points, it doesn't break the whole code. Also easier troubleshooting.

##HEATING SYSTEM CALCS
#Create system level dataframes
Heating_df = ACE_data[['Pump 4a VFD Output', 'Pump 4b VFD Output', 'Pump 4a Status', 'Pump 4b Status', 'Boiler 1% signal', 'Boiler 2% signal', 'Boiler 1 status', 'Boiler 2 status']]#Always use double [] brackets for picking the data you need
#Pull in only the heating system baseline period:
Heating_df = Heating_df.loc[start_baseline_heating:end_baseline_heating]
#Heating_df.to_csv('Heating_df.csv') #Uncomment for troubleshooting

#Calculating kW from BMS information
Heating_df['Pump 4a kW (Formula Based)'] = get_hp('Pump4a',Nameplate)*0.745699872*(Heating_df['Pump 4a VFD Output']/100)**2.5*Heating_df['Pump 4a Status']
Heating_df['Pump 4b kW (Formula Based)'] = get_hp('Pump4b',Nameplate)*0.745699872*(Heating_df['Pump 4b VFD Output']/100)**2.5*Heating_df['Pump 4b Status']
Heating_df['Boiler 1 MBtu'] = get_value('Boiler1_capacity', Boiler_Nameplate)*Heating_df['Boiler 1 status']*Heating_df['Boiler 1% signal']/(100*get_value('Boiler1_Eff', Boiler_Nameplate)) #The 100 in the denominator is to convert the % signal value
Heating_df['Boiler 2 MBtu'] = get_value('Boiler2_capacity', Boiler_Nameplate)*Heating_df['Boiler 2 status']*Heating_df['Boiler 2% signal']/(100*get_value('Boiler2_Eff', Boiler_Nameplate))

#Heating_df.to_csv('Heating_df.csv') #Uncomment for troubleshooting
Heating_df_15min = Heating_df.resample(rule='15Min').mean() #Averaging for 15 min in order to use the correlation factors
#Heating_df_15min.to_csv('Heating_df_15min.csv') #Uncomment for troubleshooting

#Calculating correlated values and adding reporting variables to dataframe
Baseline_heating['Boiler 1 MBtu'] = Heating_df_15min['Boiler 1 MBtu']
Baseline_heating['Boiler 2 MBtu'] = Heating_df_15min['Boiler 2 MBtu']
Baseline_heating['Total Boiler NG Consumption (MBtu)'] = Heating_df_15min[['Boiler 1 MBtu', 'Boiler 2 MBtu']].sum(axis=1, min_count=1)
Baseline_heating['Pump 4a kW (Correlated)'] = Heating_df_15min['Pump 4a kW (Formula Based)'] * Corr_param_df['slope'][0] + Corr_param_df['intercept'][0]
Baseline_heating['Pump 4b kW (Correlated)'] = Heating_df_15min['Pump 4b kW (Formula Based)'] * Corr_param_df['slope'][1] + Corr_param_df['intercept'][1]
Baseline_heating['Heating System kW'] = Baseline_heating[['Pump 4a kW (Correlated)', 'Pump 4b kW (Correlated)']].sum(axis=1, min_count=1) #If there are columns with NAN data then the sum won't work and the total column will also be nan. The min_count deals with this so if at least one column has a non-NaN value, the sum will be computed using the non-NaN values. The NaN values will be ignored


##AHU-19 CALCS
AHU_df = ACE_data[['AHU19 supply fan VFD output', 'AHU19 Supply fan Status', 'AHU19 Exhaust fan 1 VFD speed', 'AHU19 Exhaust fan 2 VFD speed', 'AHU19 Heat Recovery Wheel VFD', 'AHU19 Heat Recovery Wheel Status', 'AHU19 Exhaust fan CFM']]
#Pull in only the ventilation system baseline period:
AHU_df = AHU_df.loc[start_baseline_vent:end_baseline_vent]
#Calculating kW from BMS information
AHU_df['AHU 19 EF1 kW (Formula Based)'] = (get_hp('AHU19EF1', Nameplate))*0.745699872*(AHU_df['AHU19 Exhaust fan 1 VFD speed']/100)**2.5 #No status exists
AHU_df['AHU 19 EF2 kW (Formula Based)'] = (get_hp('AHU19EF2', Nameplate))*0.745699872*(AHU_df['AHU19 Exhaust fan 2 VFD speed']/100)**2.5 #No status exists
AHU_df['AHU 19 SF kW (Formula Based)'] = AHU_df['AHU19 Supply fan Status']*(get_hp('AHU19SF', Nameplate))*0.745699872*(AHU_df['AHU19 supply fan VFD output']/100)**2.5
AHU_df['AHU 19 HRW kW (Formula Based)'] = AHU_df['AHU19 Heat Recovery Wheel Status']*(get_hp('AHU19HRW', Nameplate))*0.745699872*(AHU_df['AHU19 Heat Recovery Wheel VFD']/100)**2.5
AHU_df['AHU 19 Total kW (Formula Based)'] = AHU_df[['AHU 19 EF1 kW (Formula Based)', 'AHU 19 EF2 kW (Formula Based)', 'AHU 19 SF kW (Formula Based)', 'AHU 19 HRW kW (Formula Based)']].sum(axis=1, min_count=1)
AHU_df_15min = AHU_df.resample(rule='15Min').mean()
#AHU_df.to_csv('AHU_df.csv) #Uncomment for troubleshooting

#Calculating correlated values and adding reporting variables to dataframe
Baseline_vent['AHU 19 Total kW (Correlated)'] = AHU_df_15min['AHU 19 Total kW (Formula Based)']* Corr_param_df['slope'][5] + Corr_param_df['intercept'][5]

##HRU CALCS
HRU_df = ACE_data[['HRU supply fan VFD output', 'HRU Exhaust fan VFD output', 'HRU Exhaust Fan Status', 'HRU Supply Fan Status']]
#Pull in only the ventilation system baseline period:
HRU_df = HRU_df.loc[start_baseline_vent:end_baseline_vent]
HRU_df['HRU Supply Fan kW (Formula Based)'] = HRU_df['HRU Supply Fan Status']*(get_hp('HRUSupplyFan',Nameplate))*0.745699872*(HRU_df['HRU supply fan VFD output']/100)**2.5
HRU_df['HRU Exhaust Fan kW (Formula Based)'] = HRU_df['HRU Exhaust Fan Status']*(get_hp('HRUReturnFan',Nameplate))*0.745699872*(HRU_df['HRU Exhaust fan VFD output']/100)**2.5
HRU_df['HRU Total kW (Formula Based)'] = HRU_df[['HRU Exhaust Fan kW (Formula Based)', 'HRU Supply Fan kW (Formula Based)']].sum(axis=1, min_count=1) #One DENT on all HRU so will need to correlate to total
HRU_df_15min = HRU_df.resample(rule='15Min').mean()
#HRU_df_15min.to_csv('HRU_df_15min.csv')

#Calculating correlated values and adding reporting variables to dataframe
Baseline_vent['HRU Total kW (Correlated)'] = HRU_df_15min['HRU Total kW (Formula Based)'] * Corr_param_df['slope'][4] + Corr_param_df['intercept'][4]

#CHILLED WATER SYSTEM CALCS
CHW_df = ACE_data[['Chilled water power meter', 'Pump 2a-b VFD output', 'Pump 2a status', 'Pump 2b status', 'Pump 1a feedback', 'Pump 1b feedback', 'Pump 1 VFD Signal', 'Pump 3a status', 'Pump 3b status', 'Chiller status', 'Cooling Tower Free Cool Status', 'Cooling tower fan %speed', 'Cooling tower Fan 1 Status', 'Cooling tower Fan 2 Status']]

#Calculating kW from BMS information
CHW_df['Pump 1a kW (Formula Based)'] = get_hp('Pump1a',Nameplate)*0.745699872*(CHW_df['Pump 1a feedback']/100)**2.5 #No status exists #todo: add correlation if needed
CHW_df['Pump 1b kW (Formula Based)'] = get_hp('Pump1b', Nameplate)*0.745699872*(CHW_df['Pump 1b feedback']/100)**2.5 #No status exists and does not need correlation
CHW_df['Pump 2b kW (Formula Based)'] = CHW_df['Pump 2b status']*get_hp('Pump2b',Nameplate)*0.745699872*(CHW_df['Pump 2a-b VFD output']/100)**2.5
CHW_df['Pump 2a kW (Formula Based)'] = CHW_df['Pump 2a status']*(get_hp('Pump2a',Nameplate)*0.745699872*(CHW_df['Pump 2a-b VFD output']/100)**2.5)
CHW_df['Pump 3a kW (Formula Based)'] = CHW_df['Pump 3a status']*(get_hp('Pump3a', Nameplate))*0.745699872
CHW_df['Pump 3b kW (Formula Based)'] = CHW_df['Pump 3b status']*(get_hp('Pump3a', Nameplate))*0.745699872
CHW_df['Tower Fan 1 kW (Formula Based)'] = CHW_df['Cooling tower Fan 1 Status']*(get_hp('CTFan1', Nameplate))*0.745699872*(CHW_df['Cooling tower fan %speed']/100)**2.5
CHW_df['Tower Fan 2 kW (Formula Based)'] = CHW_df['Cooling tower Fan 2 Status']*(get_hp('CTFan2', Nameplate))*0.745699872*(CHW_df['Cooling tower fan %speed']/100)**2.5
CHW_df['Chiller kW'] = CHW_df['Chiller status'] * CHW_df['Chilled water power meter'] #Todo: Seeing a bunch of negative readings will need to drop those or set it to 0

CHW_df_15min = CHW_df.resample(rule='15Min').mean()


#Calculating correlated values and adding reporting variables to dataframe
#Baseline_df['Pump 1a kW (Correlated)'] = CHW_df_15min['Pump 1a kW (Formula Based)']* Corr_param_df['slope'][x] + Corr_param_df['intercept'][x] #Todo: Add corr parameters when available
Baseline_cooling['Pump 1a kW (Formula Based)'] = CHW_df_15min['Pump 1a kW (Formula Based)']
Baseline_cooling['Pump 1b kW (Formula Based)'] = CHW_df_15min['Pump 1b kW (Formula Based)']
Baseline_cooling['Pump 2a kW (Correlated)'] = CHW_df_15min['Pump 2a kW (Formula Based)'] * Corr_param_df['slope'][2] + Corr_param_df['intercept'][2]
Baseline_cooling['Pump 2b kW (Correlated)'] = CHW_df_15min['Pump 2b kW (Formula Based)'] * Corr_param_df['slope'][3] + Corr_param_df['intercept'][3]
Baseline_cooling['Pump 3a kW (Formula Based)'] = CHW_df_15min['Pump 3a kW (Formula Based)']
Baseline_cooling['Pump 3b kW (Formula Based)'] = CHW_df_15min['Pump 3b kW (Formula Based)']
Baseline_cooling['Tower Fan 1 kW (Correlated)'] = CHW_df_15min['Tower Fan 1 kW (Formula Based)'] * Corr_param_df['slope'][6] + Corr_param_df['intercept'][6] #the only electricty consuming equipment in the CTs are the fan assemblies
Baseline_cooling['Tower Fan 2 kW (Correlated)'] = CHW_df_15min['Tower Fan 2 kW (Formula Based)'] * Corr_param_df['slope'][7] + Corr_param_df['intercept'][7]
Baseline_cooling['Chiller kW'] = CHW_df_15min['Chiller kW']
Baseline_cooling['Total CHW kW'] = Baseline_cooling[['Pump 1a kW (Formula Based)', 'Pump 1b kW (Formula Based)', 'Pump 2a kW (Correlated)', 'Pump 2b kW (Correlated)', 'Pump 3a kW (Formula Based)', 'Pump 3b kW (Formula Based)', 'Tower Fan 1 kW (Correlated)', 'Tower Fan 2 kW (Correlated)', 'Chiller kW']].sum(axis=1, min_count=1)

Baseline_heating_hourly = Baseline_heating.resample(rule='H').mean()
Baseline_heating_hourly['Total Boiler NG Consumption (MMBtu)'] = Baseline_heating_hourly['Total Boiler NG Consumption (MBtu)']/1000
Baseline_heating_hourly['Total Heating Plant Energy Consumption (MMBtu)'] = Baseline_heating_hourly['Total Boiler NG Consumption (MMBtu)'] + (Baseline_heating_hourly['Heating System kW'] * 0.003412) #converting total consumption to MMBtu

Baseline_vent_hourly = Baseline_vent.resample(rule='H').mean()
Baseline_cooling_hourly = Baseline_cooling.resample(rule='H').mean()
#Baseline_df_hourly.to_csv('Baseline_df_hourly.csv') #You know the drill

#Get weather data from Open Meteo #Todo: For future projects convert this into a function that just takes the start and end date as inputs. Maybe the variables too? (this exists in AMI Functions now, could pull over for this project)

# Setup the Open-Meteo API client with cache and retry on error.
cache_session = requests_cache.CachedSession('.cache', expire_after=-1) #Caching prevents the need for multiple API calls which is important since open meteo has a fixed number of free API calls
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Define the parameters as variables
latitude = 44.48
longitude = -73.21
hourly_variables = ["temperature_2m", "dew_point_2m", "precipitation", "weather_code"]
temperature_unit = "fahrenheit"
timezone = "America/New_York"
start_date = start_baseline_heating
end_date = end_baseline_heating

# Parameters dictionary using variables
params = {
    "latitude": latitude,
    "longitude": longitude,
    "hourly": hourly_variables,
    "temperature_unit": temperature_unit,
    "timezone": timezone,
    "start_date": start_date,
    "end_date": end_date
}

# Make the API request
responses = openmeteo.weather_api(url="https://archive-api.open-meteo.com/v1/archive", params=params)
# Process the first location.
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. Ensure the order of variables matches the request
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
hourly_weather_code = hourly.Variables(3).ValuesAsNumpy()

# Prepare hourly data with date range and assign the values
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

# Create a DataFrame from the hourly data
hourly_weather_dataframe = pd.DataFrame(data=hourly_data)

hourly_weather_df = hourly_weather_dataframe.drop(['dew_point_2m', 'precipitation', 'weather_code'], axis=1) #Dropping whatever variables are not going to be important
hourly_weather_df['date'] = pd.to_datetime(hourly_weather_df['date']) #Date from open meteo is a RangeIndex and resample only works on DatetimeIndex, TimedeltaIndex or PeriodIndex. The dt.date drops the time component otherwise resampling was not working properly.
hourly_weather_df.set_index('date', inplace=True) #Setting the date as index

#hourly_weather_df.to_csv('Hourly_weather_df.csv') #You know the drill

Baseline_heating_hourly = pd.merge(Baseline_heating_hourly, hourly_weather_df, how='outer', left_index=True, right_index=True) #Merging energy data with wetaher data
Baseline_vent_hourly = pd.merge(Baseline_vent_hourly, hourly_weather_df, how='outer', left_index=True, right_index=True) #Merging energy data with wetaher data
Baseline_cooling_hourly = pd.merge(Baseline_cooling_hourly, hourly_weather_df, how='outer', left_index=True, right_index=True) #Merging energy data with wetaher data

#Now to determine the balance point

# Grouping by day
Baseline_heating_daily = Baseline_heating_hourly.resample('D').mean()
Baseline_vent_daily = Baseline_vent_hourly.resample('D').mean()
Baseline_cooling_daily = Baseline_cooling_hourly.resample('D').mean()
#Baseline_df_daily.to_csv('Baseline_df_daily.csv') #Uncomment for troubleshooting

##Plotting to determine balance point

#Heating Balance Point
#Plotting NG consumption vs temp
plt.figure(figsize=(10, 6))
plt.scatter(Baseline_heating_daily['temperature_2m'], Baseline_heating_daily['Total Boiler NG Consumption (MBtu)'])
plt.xlabel('Average Temperature (F)')
plt.ylabel('Boiler NG Usage')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Baseline\Heating_balance_point_NG.png')
plt.close()

#Plotting Heating System kW consumption vs temp
plt.figure(figsize=(10, 6))
plt.scatter(Baseline_heating_daily['temperature_2m'], Baseline_heating_daily['Heating System kW'])
plt.xlabel('Average Temperature (F)')
plt.ylabel('Heating System kW Usage')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Baseline\Heating_balance_point_kW.png')
plt.close()

#Plotting AHU System kW consumption vs temp
plt.figure(figsize=(10, 6))
plt.scatter(Baseline_vent_daily['temperature_2m'], Baseline_vent_daily['AHU 19 Total kW (Correlated)'])
plt.xlabel('Average Temperature (F)')
plt.ylabel('AHU kW Usage')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Baseline\AHU_Temp_kW.png')
plt.close()

#Plotting HRU System kW consumption vs temp
plt.figure(figsize=(10, 6))
plt.scatter(Baseline_vent_daily['temperature_2m'], Baseline_vent_daily['HRU Total kW (Correlated)'])
plt.xlabel('Average Temperature (F)')
plt.ylabel('HRU kW Usage')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Baseline\HRU_temp_kW.png')
plt.close()

#Cooling Balance Point #todo: uncomment it out when needed
# plt.figure(figsize=(10, 6))
# plt.scatter(monthly_balance_point_df['temperature_2m'], monthly_balance_point_df['Total CHW kW'])
# plt.xlabel('Average Temperature (F)')
# plt.ylabel('Total CHW kW')
# plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\Baseline\Cooling_balance_point.png')
# plt.close()

balance_point_HDD = 45 #Todo Updated once on 03/03/25, will need to be checked after
balance_point_CDD = 75 #Todo update based on baseline data

#Calculate HDD and CDD

# # Calculate HDD (when temperature is below balance_point_HDD) #Commenting this out cause using hourly data is overestimating energy usage
# Baseline_df_hourly ['HDD'] = Baseline_df_hourly ['temperature_2m'].apply(
#     lambda temp: abs(temp - balance_point_HDD) if temp < balance_point_HDD else 0)
#
# # Calculate CDD (when temperature is above balance_point_CDD)
# Baseline_df_hourly ['CDD'] = Baseline_df_hourly['temperature_2m'].apply(
#     lambda temp: abs(temp - balance_point_CDD) if temp > balance_point_CDD else 0)
#
# Baseline_df_hourly['Total DD'] = Baseline_df_hourly[['HDD','CDD']].sum(axis=1, min_count=1)

# Calculate HDD (when temperature is below balance_point_HDD)
Baseline_heating_daily['HDD'] = Baseline_heating_daily['temperature_2m'].apply(
    lambda temp: abs(temp - balance_point_HDD) if temp < balance_point_HDD else 0)

# Calculate CDD (when temperature is above balance_point_CDD)
Baseline_heating_daily['CDD'] = Baseline_heating_daily['temperature_2m'].apply(
    lambda temp: abs(temp - balance_point_CDD) if temp > balance_point_CDD else 0)

Baseline_heating_daily['Total DD'] = Baseline_heating_daily[['HDD','CDD']].sum(axis=1, min_count=1)

#Fun methods figure
plt.scatter(Baseline_heating_daily['HDD'],Baseline_heating_daily['Total Heating Plant Energy Consumption (MMBtu)'])
plt.xlabel('HDD')
plt.ylabel('Heating System Energy Usage (MMbtu)')
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\HeatingModel_daily.png')
plt.close()

# plt.scatter(Baseline_df_hourly['CDD'],Baseline_df_hourly['Total CHW kW']) #todo: uncomment this out when needed
# plt.xlabel('Cooling Degree Days')
# plt.ylabel('Cooling System Energy Usage (kWh)')
# plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\CoolingModel.png')
# plt.close()


# List of columns to check for NaN values. Due to difference in how open meteo and ACE handle API requests, we get some additional rows where we have no ACE data which causes the regression to not work
columns_to_check = ['Total Heating Plant Energy Consumption (MMBtu)', 'AHU 19 Total kW (Correlated)',
                    'HRU Total kW (Correlated)', 'Total CHW kW']

# # Drop rows where all the specified columns have NaN values
# Baseline_df_hourly= Baseline_df_hourly.dropna(subset=columns_to_check, how='all')
# Baseline_df_hourly.index = pd.to_datetime(Baseline_df_hourly.index)
# Baseline_df_hourly = Baseline_df_hourly[(Baseline_df_hourly.index>= start_check) & (Baseline_df_hourly.index <=end_check)]

# Drop rows where all the specified columns have NaN values
Baseline_df_daily= Baseline_heating_daily.dropna(subset=columns_to_check, how='all')
Baseline_df_daily.index = pd.to_datetime(Baseline_df_daily.index)
Baseline_df_daily = Baseline_df_daily[(Baseline_df_daily.index>= start_check) & (Baseline_df_daily.index <=end_check)]

Baseline_df_daily.to_csv('Baseline_heating_daily.csv')

##Now to fit regression equations for normalization

# Fitting the models
Heating_model = LinearRegression().fit(
    Baseline_df_daily['HDD'].values.reshape(-1, 1),
    Baseline_df_daily['Total Heating Plant Energy Consumption (MMBtu)']
)

#todo: Uncomment when enough data is available

# AHU19_model = LinearRegression().fit(
#     Baseline_vent_hourly['Total DD'].values.reshape(-1, 1),
#     Baseline_vent_hourly['AHU 19 Total kW (Correlated)']
# )
#
# HRU_model = LinearRegression().fit(
#     Baseline_vent_hourly['Total DD'].values.reshape(-1, 1),
#     Baseline_vent_hourly['HRU Total kW (Correlated)']
# )

# CHW_model = LinearRegression().fit(
#     Baseline_df_hourly['CDD'].values.reshape(-1, 1),
#     Baseline_df_hourly['Total CHW kW']
# )

# Save in a .csv
with open(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Baseline_Model_Regression_Parameters.csv', 'w',
          newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Model name', 'slope', 'intercept', 'rsquared'])

    writer.writerow([
        'Heating_model',
        float(Heating_model.coef_),
        float(Heating_model.intercept_),
        Heating_model.score(
            Baseline_df_daily['HDD'].values.reshape(-1, 1),
            Baseline_df_daily['Total Heating Plant Energy Consumption (MMBtu)']
        )
    ])

    # writer.writerow([
    #     'AHU19_model',
    #     float(AHU19_model.coef_),
    #     float(AHU19_model.intercept_),
    #     AHU19_model.score(
    #         Baseline_vent_hourly['Total DD'].values.reshape(-1, 1),
    #         Baseline_vent_hourly['AHU 19 Total kW (Correlated)']
    #     )
    # ])
    #
    # writer.writerow([
    #     'HRU_model',
    #     float(HRU_model.coef_),
    #     float(HRU_model.intercept_),
    #     HRU_model.score(
    #         Baseline_vent_hourly['Total DD'].values.reshape(-1, 1),
    #         Baseline_vent_hourly['HRU Total kW (Correlated)']
    #     )
    # ])
    #
    # writer.writerow([
    #     'CHW_model',
    #     float(CHW_model.coef_),
    #     float(CHW_model.intercept_),
    #     CHW_model.score(
    #         Baseline_df_hourly['CDD'].values.reshape(-1, 1),
    #         Baseline_df_hourly['Total CHW kW']
    #     )
    # ])

#Plot regressions for checking
x = Baseline_df_daily['HDD'].values.reshape(-1, 1)
y = Baseline_df_daily['Total Heating Plant Energy Consumption (MMBtu)'].values

# Predicting y values using the fitted model
y_pred = Heating_model.predict(x)

# Plotting the data points
plt.scatter(x, y, color='blue', label='Data points')

# Plotting the regression line
plt.plot(x, y_pred, color='red', linewidth=2, label='Regression line')

# Adding labels and title
plt.xlabel('Heating Degree Days (HDD)')
plt.ylabel('Total Heating Plant Energy Consumption (MMBtu)')
plt.title('Linear Regression: HDD vs. Heating Energy Consumption')

# Adding a legend
plt.legend()

# Showing the plot
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\HeatingModel_with regression line.png')
plt.close()