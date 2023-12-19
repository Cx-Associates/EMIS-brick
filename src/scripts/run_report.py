'''

You've trained a baseline model, and now you're running the report.
'''
# load a prior project and modelset.
project_filepath = None
modelset_filepath = None

# set filepath for single .csv report
single_report_filepath = None

# set filepath for ledger .csv file to update with this report
ledger_filepath = None

# set reporting period for this report
reporting_period = ('2023-12-12', '2023-12-19')

# load the project and the modelset based on specified filepaths above
project = load_from_pickle()
modelset = load_from_pickle()

# prints useful modelset attributes
modelset.hello()

# choose models to use for report
model1 = modelset.systems['heating_system'].energy_models['TODTweekend']
model2 = modelset.equipment['chiller'].energy_models['TOWT']

modelset.report(
    models=[model1, model2]
)
