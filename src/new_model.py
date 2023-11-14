"""Modeling and reporting script, likely for a new project, or for a run for which you don't need to load a previous
project.

"""
from utils import Project, EnergyModelset

project = Project(
    name='Main Street Landing',
    location=(44.48, -73.21),
    time_frame=('past week'),
)

modelset = EnergyModelset(
    project,
    models=[
        'heating_system',
        'chw_system',
        'chiller',
    ]
)

# We can use pump speed (P4a and P4b) as a proxy for flow, and we know delta T between supply and return. We can then
# set up a proxy energy consumption value as e = m*c*deltaT. Calibrate this against monthly fuel consumption data in
# the future if we want. It doesn't need to be calibrated for energy management. The biggest missing piece here: is
# there a characteristic equation (can we look at pump curves) to describe how well pump speed approximates total
# mass flow, knowing that the system is set to maintain dP setpoint and radiator valves are opening and closing to
# raise/drop system pressure.

modelset.heating_system.set(
    predict=['boilers', 'pumps', 'deltaT'],
    functionOf=['TOWT']
)

modelset.chw_system.set(
    predict=[''],
    functionOf=['TOWT', 'occupancy']
)

modelset.chiller.set(
    ['chiller_power_meter'],
    functionOf=['TOWT']
)