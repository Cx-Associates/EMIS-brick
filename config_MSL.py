'''Config file for

'''

config_dict = {
    'name': 'Main Street Landing',
    'location': (44.48, -73.21),
}

# feature engineering
def heating_system_Btus(df):
    '''

    :param df:
    :return:
    '''
    df['deltaT'] = df['heating_sysytem_return_water__heating']