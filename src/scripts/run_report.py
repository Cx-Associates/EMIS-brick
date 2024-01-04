'''

You've trained a baseline model, and now you're running the report. The result of the report is exported / updated
.csv files with model data.
'''
from src.utils import load_modelset, ModelPlus

# load a prior modelset (which includes a project as an attribute)
# must be connected to CxA VPN to access this exported model. See 'new_model.py' to run/export you own model
filepath_modelset = r"F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\exported modelsets\modelset_Main Street " \
                    r"Landing--msl_heating-cooling.ttl--2024-01-04_08h36m20s.bin"

# set filepath for single .csv report
dir_single_report = r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\csv_reports'

# set filepath for ledger .csv 'ledger' file to update with this report
filepath_ledger = r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\report_ledger.csv'

# load the project and the modelset based on specified filepaths above
modelset = load_modelset(filepath_modelset)

# set reporting period for this report
reporting_period = ('2023-12-23', '2023-12-31')
modelset.project.set_time_frames(reporting=reporting_period)

# prints useful modelset attributes
modelset.whosthere()

# the easiest way to pass project-related parameters into the report function is by using this ModelPlus class
# this is also where we select the models we want to see in the report
model1 = ModelPlus(
    modelset,
    entity_name='heating_system',
    model_name='TODTweekend'
)
model2 = ModelPlus(
    modelset,
    entity_name='chiller',
    model_name='TOWT'
)

# now, run the report function, passing in the ModelPlus objects we just instantiated
modelset.report(
    export_dir=dir_single_report,
    models=[model1, model2],
    ledger_filepath=filepath_ledger
)
