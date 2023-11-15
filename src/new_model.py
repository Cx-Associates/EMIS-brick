"""Modeling and reporting script, likely for a new project, or for a run for which you don't need to load a previous
project.

"""
from utils import Project, EnergyModelset

project = Project(
    name='Main Street Landing',
    location=(44.48, -73.21),
)
project.set_time_frames(
    baseline=('2023-11-08T00:00:00', '2023-11-15T00:00:00'),
)

graph_path = 'src/brick_models/msl.ttl'
project.set_metadata(graph_path)

modelset = EnergyModelset(
    project,
    models=[
        'heating_system',
        'chw_system',
        'chiller',
    ]
)
modelset.get_data()

# We can use pump speed (P4a and P4b) as a proxy for flow, and we know delta T between supply and return. We can then
# set up a proxy energy consumption value as e = m*c*deltaT. Calibrate this against monthly fuel consumption data in
# the future if we want. It doesn't need to be calibrated for energy management. The biggest missing piece here: is
# there a characteristic equation (can we look at pump curves) to describe how well pump speed approximates total
# mass flow, knowing that the system is set to maintain dP setpoint and radiator valves are opening and closing to
# raise/drop system pressure.

modelset.heating_system.train(
    predict=['boilers', 'pumps', 'dT'],
    functionOf=['TOWT']
)

modelset.chw_system.train(
    predict=[''],
    functionOf=['TOWT', 'occupancy']
)

modelset.chiller.train(
    ['chiller_power_meter'],
    functionOf=['TOWT']
)