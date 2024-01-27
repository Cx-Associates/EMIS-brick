"""Modeling and reporting script, likely for a new project, or for a run for which you don't need to load a previous
project.

As a result, the model is exported as a .bin file (using Python pickle)

"""
import os

from src.utils import Project, EnergyModelset

# import the configuration dictionary, which stores project-specific attributes
from config_MSL import config_dict

# create an instance of the project class, giving it a name and location coordinates
project = Project(
    name=config_dict['name'],
    location=config_dict['location'],
)

# set time frames for the project. #ToDo: build the option for setting baselines for individual models
project.set_time_frames(
    baseline=('2023-11-10', '2023-12-23'),
    # reporting=('2023-12-26', '2024-01-03')
)

# give filepath for brick model .ttl file, and load it into the project
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

## 'cast' models as any of TOWT, TODTweekend, TODT (for now)
modelset.systems['heating_system'].feature_engineering()
modelset.equipment['chiller'].feature_engineering()
modelset.set_models([
    ('heating_system', 'TOWT'),
    ('heating_system', 'TODTweekend'),
    ('chiller', 'TOWT'),
    ('chiller', 'TODTweekend'),
    # ('sgndoigdsion', 'fsanio'),
])

# train models within modelset
# ToDo: modelset.systems.train_all()?
modelset.systems['heating_system'].train()
modelset.equipment['chiller'].train()

# create two model variables and show some plots
model1 = modelset.systems['heating_system'].energy_models['TODTweekend']
model2 = modelset.equipment['chiller'].energy_models['TOWT']
for model in [model1, model2]:
    model.timeplot(weather_data=project.weather_data)
    model.scatterplot()
    # model.dayplot(weather_data=project.weather_data)
    model.dayplot()

# export modelset
dir_modelset = os.path.join('F:', 'PROJECTS', '1715 Main Street Landing EMIS Pilot', 'code', 'exported modelsets')
modelset.export(dir_modelset)


# modelset.report(
#     models=[model1, model2]
# )

pass