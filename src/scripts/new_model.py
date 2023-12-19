"""Modeling and reporting script, likely for a new project, or for a run for which you don't need to load a previous
project.

"""
import pickle

from src.utils import Project, EnergyModelset
from config_MSL import config_dict, heating_system_Btus

# # create an instance of the project class, giving it a name and a location
# project = Project(
#     name=config_dict['name'],
#     location=config_dict['location'],
# )
#
# # set the project baseline period
# project.set_time_frames(
#     baseline=('2023-11-08', '2023-12-09'),
#     reporting=('2023-12-10', '2023-12-18')
# )
#
# # set filepath for brick model .ttl file, and load it into the project
# graph_path = 'brick_models/msl_heating-cooling.ttl'
# project.load_graph(graph_path)
#
# # create an instance of the energy modelset class and designate the systems for which to create individual energy models
# modelset = EnergyModelset(
#     project,
#     systems=[
#         'heating_system',
#         # 'chilled_water_system',
#     ],
#     equipment=[
#         'chiller'
#     ]
# )
#
# # get weather data, then get relevant building timeseries data
# project.get_weather_data()
# modelset.get_data()
#
# # custom feature engineering for heating system
# ## 'cast' models as any of TOWT, TODTweekend, TODT (for now)
# modelset.systems['heating_system'].feature_enginering()
# modelset.equipment['chiller'].feature_enginering()
# modelset.set_models([
#     ('heating_system', 'TOWT'),
#     ('heating_system', 'TODTweekend'),
#     ('chiller', 'TOWT'),
#     ('chiller', 'TODTweekend'),
#     # ('sgndoigdsion', 'fsanio'),
# ])
#
# # train models within modelset
# modelset.systems['heating_system'].train()
# modelset.equipment['chiller'].train()
#
# # create two model variables and show some plots
# model1 = modelset.systems['heating_system'].energy_models['TODTweekend']
# model2 = modelset.equipment['chiller'].energy_models['TOWT']
# for model in [model1, model2]:
#     model.timeplot(weather_data=project.weather_data)
#     model.scatterplot()
#     # model.dayplot(weather_data=project.weather_data)
#     model.dayplot()
#
# # pickling
# with open('project.bin', 'wb') as f:
#     pickle.dump(project, f)
# with open('modelset.bin', 'wb') as f:
#     pickle.dump(modelset, f)

with open('project.bin', 'rb') as f:
    project = pickle.load(f)
with open('modelset.bin', 'rb') as f:
    modelset = pickle.load(f)

model1 = modelset.systems['heating_system'].energy_models['TODTweekend']
model2 = modelset.equipment['chiller'].energy_models['TOWT']

modelset.report(
    models=[model1, model2]
)

pass