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
    baseline=('2023-11-08T00:00:00', '2023-12-03T00:00:00'),
)

# set filepath for brick model .ttl file, and load it into the project
graph_path = 'brick_models/msl_heating-only.ttl'
project.load_graph(graph_path)

# create an instance of the energy modelset class and designate the systems for which to create individual energy models
modelset = EnergyModelset(
    project,
    systems=[
        'heating_system',
        # 'chilled_water_system',
        # 'chiller',
    ]
)

# get data for each modelset (for each designated time frame)
modelset.get_data()

# We can use pump speed (P4a and P4b) as a proxy for flow, and we know delta T between supply and return. We can then
# set up a proxy energy consumption value as e = m*c*deltaT. Calibrate this against monthly fuel consumption data in
# the future if we want. It doesn't need to be calibrated for energy management. The biggest missing piece here: is
# there a characteristic equation (can we look at pump curves) to describe how well pump speed approximates total
# mass flow, knowing that the system is set to maintain dP setpoint and radiator valves are opening and closing to
# raise/drop system pressure.

# custom feature engineering for heating system
modelset.systems['heating_system'].add_features()

modelset.systems['heating_system'].train(
    predict=['pseudo_Btus'],
    functionOf=['TOWT']
)

# modelset.systems['chilled_water_system'].train(
#     predict=[''],
#     functionOf=['TOWT', 'occupancy']
# )
#
# modelset.systems['chiller'].train(
#     ['chiller_power_meter'],
#     functionOf=['TOWT']
# )