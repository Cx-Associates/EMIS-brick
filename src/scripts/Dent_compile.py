"""
Get data from Ace and dents
Combine data
Be prepared to answer any questions that might come up from this
"""

#Copied from new_model - probably don't need all this, but it works...
import pickle

from src.utils import Project, EnergyModelset, join_csv_data
from config_MSL import config_dict, heating_system_Btus

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
project.get_weather_data()
modelset.get_data()

#end copied from new_model

dentdatapath=r"F:/PROJECTS/1715 Main Street Landing EMIS Pilot/data/Dent data pull, 2024-01-05/raw data\\E876C-01.csv"

#join_csv_data(modelset.systems.'heating_system'.dataframe,dentdatapath)
