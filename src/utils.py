"""

"""
import pandas as pd

from subrepos.energy_models.src.utils import Model, TOWT
from subrepos.energy_models.src.open_meteo import open_meteo_get
from subrepos.brickwork.utils import BrickModel


class TimeFrame():
    def __init__(self, arg):
        if isinstance(arg, tuple):
            self.tuple = arg
            # ToDo: check for correct api formatting
        elif isinstance(arg, str):
            pass
            # ToDO: auto-parse to ensure this is API-friendly
            # ToDo: where should TZ localization take place? Project has coordinates ...

class Project():
    """Project class.

    """
    def __init__(self, **kwargs):
        self.name = None
        self.location = None
        self.brick_model = None
        self.weather_data = None  # pandas.Series
        self.__dict__.update(kwargs.copy())
        if isinstance(self.location, str):
            pass #ToDo: add code to resolve lat, long as a function of place, e.g. google geocode REST api

    def load_graph(self, model_path):
        """

        :param model_path:
        :return:
        """
        self.brick_model = BrickModel(model_path)
        self.graph_filepath = model_path

    def set_time_frames(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        min_date, max_date = None, None
        time_frames = {}
        if "baseline" in kwargs.keys():
            str_ = kwargs['baseline']
            baseline = TimeFrame(str_)
            time_frames.update({'baseline': baseline})
            min_date, max_date = baseline.tuple[0], baseline.tuple[1]
        if "performance" in kwargs.keys():
            pass #ToDo: refactor above lines and repeat for performance and report
        if "report" in kwargs.keys():
            pass
        if None in [min_date, max_date]:
            raise('No time frames passed. Need any of: baseline, performance, or report kwargs.')
        for key, value in time_frames.items():
            start, end = value.tuple[0], value.tuple[1]
            if start < min_date:
                min_date = start
            if end > max_date:
                max_date = end
        total = TimeFrame((min_date, max_date))
        time_frames.update({'total': total})
        self.time_frames = time_frames

    def join_weather_data(self, df, feature='temperature_2m'):
        """

        :param df: any time-series dataframe with a datetimeindex
        :return df: resampled to hourly
        """
        try:
            lat, long = self.location[0], self.location[1]
        except NameError:
            raise Exception(f'project {self.name} must have "location" attribute in order to get weather data.')
        df_ = df.resample('h').mean()
        start, end = df_.index[0], df_.index[-1]
        s_temp = open_meteo_get((lat, long), (start, end), feature)
        df_ = pd.concat([df_, s_temp], axis=1)

        return df_

    def get_weather_data(self, feature='temperature_2m'):
        """

        :param feature:
        :return:
        """
        try:
            lat, long = self.location[0], self.location[1]
        except NameError:
            raise Exception(f'project {self.name} must have "location" attribute in order to get weather data.')
        start, end = self.time_frames['total'].tuple[0], self.time_frames['total'].tuple[1]
        s_temp = open_meteo_get((lat, long), (start, end), feature)
        self.weather_data = s_temp

class EnergyModelset():
    """

    """
    def __init__(self, project, systems, time_frames=None):
        self.project = project
        if time_frames is None:
            self.time_frames = project.time_frames
        self.systems = {}
        for system_name in systems:
            instance = EnergyModel(
                system_name,
                project
            )
            self.systems.update({system_name: instance})

    def get_data(self):
        """For each system in the modelset (for each individual model), get timeseries data of all the entities in
        the system.

        :return:
        """
        for name, energy_model_instance in self.systems.items():
            energy_model_instance.get_data()

    def set_model_type(self, dict_):
        for model_name, model_type in dict_.items():
            if model_type == 'TOWT':
                train_start = self.project.time_frames['baseline'].tuple[0]
                train_end = self.project.time_frames['baseline'].tuple[1]
                towt = TOWT(self.systems[model_name], train_start=train_start, train_end=train_end)
                df = towt.add_TOWT_features(self.project.weather_data, temp_col='temperature_2m')
                towt.X.data = df
                self.systems[model_name] = towt


class EnergyModel(Model):
    """

    """
    def __init__(self, name, project, time_frames=None):
        super().__init__()
        self.name = name
        self.project = project
        if time_frames is None:
            self.time_frames = project.time_frames
        res = self.check_system(name) #todo: binds energy model to system, correct?
        self.system_entities = res


    def check_system(self, name=None):
        brick_model = self.project.brick_model
        res = brick_model.get_entities(name)
        if len(res.entities_list) == 0:
            raise Exception(f"Couldn't find an entity named {name} in the loaded graph.")
        elif len(res.entities_list) > 1:
            raise Exception(f"Found more than one entity named {name} in the loaded graph. Each system in the "
                            f"modelset must have a unique name; otherwise you should either modify the graph (make a "
                            f"new, abritrary system) or modify this function to filter the graph query based on more "
                            f"than simply the name of a system.")
        else:
            print(f"Found entity named {name} in graph.")
        system_entities = brick_model.get_entities_of_system(name)
        return system_entities

    def get_data(self):
        """Get timeseries data for each system in the model.

        :param project:
        :return:
        """
        #ToDo: may, in this method, want to check if there are already specified time-series for model data,
        # rather than returning all timeseries of a system (which is fine for now_
        time_frame = self.project.time_frames['total'].tuple
        df = self.project.brick_model.get_system_timeseries(
            self.name, time_frame
        )
        self.dataframe = df

    def add_system_features(self):
        """

        :return:
        """
        if self.name == 'heating_system':
            pumps_df = None
            temps_df = None
            for entity in self.project.brick_model.systems['heating_system'].entities_list:
                if entity.brick_class == 'Pump':
                    for timeseries_response in entity.last_response.values():
                        if timeseries_response.brick_class == 'Speed_Sensor':
                            if pumps_df is None:
                                pumps_df = timeseries_response.data
                            else:
                                pumps_df = pd.concat([pumps_df, timeseries_response.data], axis=1)
                elif entity.brick_class == 'Supply_Hot_Water':
                    for timeseries_response in entity.last_response.values():
                        if timeseries_response.brick_class == 'Temperature_Sensor':
                            if temps_df is None:
                                temps_df = timeseries_response.data
                            else:
                                temps_df = pd.concat([temps_df, timeseries_response.data], axis=1)
                elif entity.brick_class == 'Return_Hot_Water':
                    for timeseries_response in entity.last_response.values():
                        if timeseries_response.brick_class == 'Temperature_Sensor':
                            if temps_df is None:
                                temps_df = timeseries_response.data * -1
                            else:
                                new_data = timeseries_response.data * -1
                                temps_df = pd.concat([temps_df, new_data], axis=1)
            pumps_total_df = pumps_df.sum(axis=1)
            dTemp_df = temps_df.sum(axis=1)
            psuedo_Btus_df = pumps_total_df * dTemp_df
            self.Y.data = psuedo_Btus_df