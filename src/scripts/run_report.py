'''

You've trained a baseline model, and now you're running the report.
'''
from src.utils import load_modelset

# load a prior modelset (which includes project as an attribute)
modelset_filepath = r"F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\exported modelsets\modelset_Main Street " \
                    r"Landing--msl_heating-cooling.ttl--2023-12-22-09h19m42s.bin"

# set filepath for single .csv report
single_report_filepath = None

# set filepath for ledger .csv file to update with this report
ledger_filepath = None

# set reporting period for this report
reporting_period = ('2023-12-12', '2023-12-21')

# load the project and the modelset based on specified filepaths above
modelset = load_modelset(modelset_filepath)

# prints useful modelset attributes
modelset.whosthere()

# choose models to use for report
model1 = modelset.systems['heating_system'].energy_models['TODTweekend']
model2 = modelset.equipment['chiller'].energy_models['TOWT']

modelset.report(
    models=[model1, model2]
)
