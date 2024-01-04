'''Config file for

'''

config_dict = {
    'name': 'Main Street Landing',
    'location': (44.48, -73.21),
}


exceptions = {
    'chiller': {
        'kW': {
            ('2023-12-12', '2024-02-01'),
            ('2024-02-6', ''),
        }
    }
}


# feature engineering parameters if needed
def heating_system_Btus(df):
    '''

    :param df:
    :return:
    '''
    df['deltaT'] = df['heating_sysytem_return_water__heating']
    #ToDo: this is incomplete. not currently needed?