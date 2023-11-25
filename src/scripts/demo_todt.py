"""

"""
from src.utils import Project
from subrepos.energy_models.src.utils import TODTweekend
from copy import deepcopy

project = Project(
    name='Main Street Landing',
    location=(44.48, -73.21),
)
project.set_time_frames(
    baseline=('2023-11-08T00:00:00', '2023-11-25T00:00:00'),
)
time_frame = project.time_frames['baseline']
graph_path = 'brick_models/msl_heating-only.ttl'
project.load_graph(graph_path)

# get heating system timeseries
heating_system = project.brick_model.get_entities_of_system('heating_system')
for entity in heating_system.list:
    entity.get_all_timeseries(time_frame)
df = heating_system.join_last_response()

# join weather data with heating system data
df = project.join_weather_data(df)

# feature engineering
df.columns = [
    'p4a speed', 'p4b speed', 'return temp', 'supply temp', 'OAT'
]
df_features = df[['OAT']]
df_features['pump speed'] = df['p4a speed'] + df['p4b speed']
df_features['delta T'] = df['supply temp'] - df['return temp']

## new dataframe with btus proxy calc
df_energy = df_features[['OAT']]
df_energy['btus'] = df_features['pump speed'] * df_features['delta T']

# make energy model from the dataframe
energy_model = TODTweekend(
    df_energy,
    Y_col='btus',
    X_col='OAT'
)
energy_model.train(bins=None)

energy_model_daily = deepcopy(energy_model)
energy_model_daily.Y.test = energy_model_daily.Y.test.resample('d').sum()
energy_model_daily.y.test = energy_model_daily.y.test.resample('d').sum()
energy_model_daily.score()

pass
