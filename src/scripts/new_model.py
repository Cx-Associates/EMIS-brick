"""Modeling and reporting script, likely for a new project, or for a run for which you don't need to load a previous
project.

"""
from src.utils import Project, EnergyModelset
from config_MSL import config_dict, heating_system_Btus

# create an instance of the project class, giving it a name and a location
project = Project(
    name=config_dict['name'],
    location=config_dict['location'],
)

# set the project baseline period
project.set_time_frames(
    baseline=('2023-11-08', '2023-12-12'),
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

# custom feature engineering for heating system
## 'cast' models as any of TOWT, TODTweekend, TODT (for now)
modelset.systems['heating_system'].add_model_features()
modelset.set_models([
    ('heating_system', 'TOWT'),
    ('heating_system', 'TODTweekend'),
    ('chiller', 'TOWT'),
    ('chiller', 'TODTweekend'),
    ('sgndoigdsion', 'fsanio'),
])
modelset.systems['heating_system'].train()
df = modelset.systems['heating_system'].energy_models['TODTweekend']
df.timeplot(weather=True)
df.scatterplot()
df.dayplot()
pass

# modelset.systems['chilled_water_system'].train(
#     predict=[''],
#     functionOf=['TOWT', 'occupancy']
# )
#
# modelset.systems['chiller'].train(
#     ['chiller_power_meter'],
#     functionOf=['TOWT']
# )