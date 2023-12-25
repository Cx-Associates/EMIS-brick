'''

You've trained a baseline model, and now you're running the report.
'''
from src.utils import load_modelset

# load a prior modelset (which includes project as an attribute)
# must be connected to CxA VPN to access this exported model. See 'new_model.py' to run/export you own model
filepath_modelset = r"F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\exported modelsets\modelset_Main Street " \
                    r"Landing--msl_heating-cooling.ttl--2023-12-22-09h19m42s.bin"

# set filepath for single .csv report
dir_single_report = r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\csv_reports'

# set filepath for ledger .csv 'ledger' file to update with this report
filepath_ledger = r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\report_ledger.csv'

# load the project and the modelset based on specified filepaths above
modelset = load_modelset(filepath_modelset)

# set reporting period for this report
reporting_period = ('2023-12-14', '2023-12-24')
modelset.project.set_time_frames(reporting=reporting_period)

# prints useful modelset attributes
modelset.whosthere()

# choose models to use for report
model1 = modelset.systems['heating_system'].energy_models['TODTweekend']
model2 = modelset.equipment['chiller'].energy_models['TOWT']

modelset.report(
    dir=dir_single_report,
    models=[model1, model2],
    ledger_filepath=filepath_ledger
)
