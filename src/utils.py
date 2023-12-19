"""

"""
import pandas as pd

from subrepos.energy_models.src.utils import TOWT, TODT
from subrepos.energy_models.src.apis.open_meteo import open_meteo_get
from subrepos.brickwork.utils import BrickModel

def resample_and_join(list_):
    """ToDO: spruce up this function with some automated interval detection. For now it just resamples to hourly

    :param list:
    :return:
    """
    resampled_list = [s.resample('h').mean() for s in list_]
    df = pd.concat(resampled_list, axis=1).dropna()
    return df

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
        self.brick_graph = None
        self.weather_data = None  # pandas.Series
        self.__dict__.update(kwargs.copy())
        if isinstance(self.location, str):
            pass #ToDo: add code to resolve lat, long as a function of place, e.g. google geocode REST api

    def load_graph(self, model_path):
        """

        :param model_path:
        :return:
        """
        self.brick_graph = BrickModel(model_path)
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
    def __init__(
            self,
            project,
            systems,
            equipment,
            time_frames=None
    ):
        self.project = project
        if time_frames is None:
            self.time_frames = project.time_frames
        self.systems = {}
        self.equipment = {}
        for system_name in systems:
            model_instance = System(
                system_name,
                project
            )
            self.systems.update({system_name: model_instance})
        for equipment_name in equipment:
            model_instance = Equipment(
                equipment_name,
                project
            )
            self.equipment.update({equipment_name: model_instance})


    def get_data(self):
        """For each system in the modelset (for each individual model), get timeseries data of all the entities in
        the system.

        :return:
        """
        for name, energy_model_instance in self.systems.items():
            energy_model_instance.get_data()
        for name, energy_model_instance in self.equipment.items():
            energy_model_instance.get_data()

    def set_models(self, list_):
        system = None
        for tuple in list_:
            system_name, model_type = tuple[0], tuple[1]
            try:
                system = self.systems[system_name]
            except AttributeError:
                msg = f'System {system_name} not found in brick graph.'
            df = resample_and_join([system.Y_data, self.project.weather_data])
            if model_type == 'TOWT':
                towt = TOWT(
                    df,
                    Y_col='pseudo_Btus'
                )
                towt.add_TOWT_features(df, temp_col='temperature_2m')
                system.energy_models.update({'TOWT': towt})
            elif model_type == 'TODTweekend':
                todtweekend = TODT(
                    df,
                    Y_col='pseudo_Btus',
                    weekend=True
                )
                todtweekend.add_TODT_features(df, temp_col='temperature_2m')
                system.energy_models.update({'TODTweekend': todtweekend})
            else:
                msg = f'Cannot instantiate a {model_type} model for {system_name} because that model type is not yet ' \
                      f'configured.'
                raise Exception(msg)


class GraphEntity():
    """A system should be a set of physical equipment, like pumps and boilers for a hot water system. For this
    purpose, the system should not include sensors, meters, or other data-related items. A chilled water system
    might comprise a chiller, pumps, and the chilled water itself. It might or might not include elements of the
    condenser water system.

    """
    def __init__(self, name, project, time_frames=None):
        self.name = name
        self.project = project
        if time_frames is None:
            self.time_frames = project.time_frames #todo: time frames should be project property. change it and run
            # thru debugger and see what happens
        res = self.check_graph_for_entity(name) #todo: binds energy model to entity, correct?
        self.entity = res
        self.Y_data = None
        self.energy_models = {}

    def check_graph_for_entity(self, name=None, brick_class=None):
        brick_graph = self.project.brick_graph
        res = brick_graph.get_entities(name, brick_class)
        if len(res.entities_list) == 0:
            raise Exception(f"Couldn't find an entity named {name} in the loaded graph.")
        elif len(res.entities_list) > 1:
            str_ = f"Found more than one entity named {name} in the loaded graph. Each system or equipment in  the " \
                   f"modelset must have a unique name if it's going to serve as the basis for an energy model (for " \
                   f"now); otherwise you should either modify the graph or modify this function to filter the  graph " \
                   f"query based on more than simply the name of an equipment or system."
            raise Exception(str_)
        else:
            print(f"Found entity named {name} in graph. \n")

        return res

    def get_data(self):
        """Get timeseries data for each system in the model.

        :param project:
        :return:
        """
        #ToDo: may, in this method, want to check if there are already specified time-series for model data,
        # rather than returning all timeseries of a system (which is fine for now_
        time_frame = self.project.time_frames['total'].tuple
        df = self.project.brick_graph.get_system_timeseries(
            self.name, time_frame
        )
        self.dataframe = df

    def add_model_features(self):
        """

        :return:
        """
        if self.name == 'heating_system':
            pumps_df = None
            temps_df = None
            for entity in self.project.brick_graph.systems['heating_system'].entities_list:
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
            pseudo_Btus_df = pumps_total_df * dTemp_df
            self.Y_data = pseudo_Btus_df
            self.Y_data.name = 'pseudo_Btus'

    def train(self):
        for model_name, model in self.energy_models.items():
            if not 'baseline' in model.time_frames.keys():
                # unless the energy model itself has been given a baseline explicitly, at this point (for training)
                # it needs to assume the baseline period of the project as a whole.
                model.time_frames.update({'baseline': self.project.time_frames['baseline']})
            model.train()


class System(GraphEntity):
    """

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_entities = self.project.brick_graph.get_entities_of_system(self.name)

class Equipment(GraphEntity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_data(self):
        """

        :return:
        """
        time_frame = self.project.time_frames['total'].tuple
        entity = self.entity.entities_list[0]  #ToDo: change this attribute in the parent-child class framework. sloppy.
        entity.get_all_timeseries(time_frame)
        df = entity.join_last_response()
        self.dataframe = df


#ToDo: make a new class called System??