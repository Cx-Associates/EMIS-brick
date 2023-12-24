"""This script is intended to show simply how to retrieve a timeseries for a given system or equipment

"""
from src.utils import Project

# initiate project and pass in config variables
project = Project(name='Main Street Landing')

# make a time frame tuple for baseline
baseline_time_frame = ('2023-11-08T00:00:00', '2023-11-18T00:00:00')

# set the project's time frame and load the brick graph
project.set_time_frames(
    baseline=baseline_time_frame,
)
graph_path = 'brick_models/msl_heating-only.ttl'
project.load_graph(graph_path)

# get timeseries data by equipment type
pumps = project.brick_graph.get_entities(brick_class='Pump')
for pump in pumps.list_:
    pump.get_all_timeseries(baseline_time_frame)
df = pumps.join_last_response()

# get timeseries data for whole heating system
heating_system = project.brick_graph.get_entities_of_system('heating_system')
for entity in heating_system.list_:
    entity.get_all_timeseries(baseline_time_frame)
df_heating_system = heating_system.join_last_response()

# is there a boiler in the brick graph?
boiler = project.brick_graph.get_entities(brick_class='boiler')

