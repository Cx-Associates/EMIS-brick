"""

"""
import os
import pickle
from pprint import pprint

import pandas as pd
from datetime import datetime as dt

from subrepos.energy_models.src.utils import TOWT, TODT
from subrepos.energy_models.src.apis.open_meteo import open_meteo_get
from subrepos.brickwork.utils import BrickModel, TimeseriesResponse


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
        self.time_frames = {}
        self.__dict__.update(kwargs.copy())
        if isinstance(self.location, str):
            pass #ToDo: add code to resolve lat, long as a function of place, feature_engineering.g. google geocode REST api

    def load_graph(self, model_path):
        """

        :param model_path:
        :return:
        """
        self.brick_graph = BrickModel(model_path)
        self.graph_filepath = model_path


    def set_time_frames(self, **kwargs):
        """

        :param kwargs: Must be one of the following: 'baseline', 'performance', 'reporting', or 'total'
        :return:
        """
        min_date, max_date = None, None
        if self.time_frames:
            time_frames = self.time_frames
        else:
            time_frames = {}
        for key in ['baseline', 'performance', 'reporting', 'total']:
            if key in kwargs.keys():
                str_ = kwargs[key]
                tuple_ = TimeFrame(str_)
                time_frames.update({key: tuple_})
        for key, value in time_frames.items():
            start, end = value.tuple[0], value.tuple[1]
            if min_date is None:
                min_date = start
            elif start < min_date:
                min_date = start
            if max_date is None:
                max_date = end
            elif end > max_date:
                max_date = end
        if None in [min_date, max_date]:
            msg = "No time frames passed. Need any of: 'baseline', 'performance', 'reporting', or 'total' in kwargs."
            raise Exception(msg)
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

def load_modelset(filepath):
    '''Complement to export function; use pickle to load

    :param dir_:
    :return:
    '''
    with open(filepath, 'rb') as f:
        modelset_object = pickle.load(f)

        return modelset_object

def formatted_now():
    now = dt.now().strftime(f'%Y-%m-%d-%H'+'h'+'%M'+'m'+'%S'+'s')
    return now

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


    def get_data(self, time_frame='total'):
        """For each system in the modelset (for each individual model), get timeseries data of all the entities in
        the system.

        :return:
        """
        for name, entity in self.systems.items():
            entity.get_data(time_frame)
        for name, entity in self.equipment.items():
            entity.get_data(time_frame)

    def set_models(self, list_):
        entity = None
        for tuple in list_:
            entity_name, model_type = tuple[0], tuple[1]
            try:
                entity = self.systems[entity_name]
            except KeyError:
                try:
                    entity = self.equipment[entity_name]
                except KeyError:
                    msg = f'Entity {entity_name} not found as a system or equipment attribute of EnergyModelset ' \
                          f'instance.'
                    raise Exception(msg)
            if self.project.weather_data is None:
                self.project.get_weather_data()
            df = resample_and_join([entity.Y_series, self.project.weather_data])
            if model_type == 'TOWT':
                towt = TOWT(
                    df,
                    Y_col=entity.Y_series.name,
                )
                towt.add_TOWT_features(df, temp_col='temperature_2m')
                entity.energy_models.update({'TOWT': towt})
            elif model_type == 'TODTweekend':
                todtweekend = TODT(
                    df,
                    Y_col=entity.Y_series.name,
                    weekend=True
                )
                todtweekend.add_TODT_features(df, temp_col='temperature_2m')
                entity.energy_models.update({'TODTweekend': todtweekend})
            else:
                msg = f'Cannot instantiate a {model_type} model for {entity_name} because that model type is not yet ' \
                      f'configured.'
                raise Exception(msg)

    def export(self, dir_):
        '''Uses pickle to write self object to local directory

        :return:
        '''
        project_name = self.project.name
        graph_name = self.project.graph_filepath.rsplit('/')[-1]
        now = formatted_now()
        filename = f'modelset_{project_name}--{graph_name}--{now}.bin'
        #ToDO: '.ttl' in graph_name might throw off future functionality, see about replacing with underscore
        filepath = os.path.join(dir_, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f'Exported modelset to {filepath}.')

    def report(self, export_dir, models, ledger_filepath=None):
        """

        :param export_dir: directory for report (.csv) file export
        :param models: (list): must be instances of ModelPlus class.
        :param ledger_filepath: filepath to a .csv file to which this report's metrics will be added as a new row.
        :return:
        """
        try:
            time_frame = self.project.time_frames['reporting']
        except KeyError:
            msg = 'Reporting period designation not found in project. Run set_time_frames on the project and set the ' \
                  'baseline period.'
            raise Exception(msg)
        # get timeseries data from relevant entities and run feature engineering
        self.get_data(time_frame='reporting')
        for name, object in self.systems.items():
            object.feature_engineering()
            for model in object.energy_models.values():
                model.Y.pred = object.Y_series
        for name, object in self.equipment.items():
            object.feature_engineering()
            for model in object.energy_models.values():
                model.Y.pred = object.Y_series
        project_name = self.project.name
        graph_name = self.project.graph_filepath.rsplit('/')[-1]
        df = None
        for model_plus in models:
            model = model_plus.model
            # first, make sure the model has the project's location, in case location is needed to request
            # additional weather data for this prediction
            if not model.location:
                model.location = self.project.location
            # now run the prediction. if the model is TODT or TOWT, it will ask for weather data it doesn't have.
            model.predict(time_frame)
            model.reporting_metrics()
            model.report.update({
                'baseline_period': str(self.project.time_frames['baseline'].tuple),
                'reporting_period': str(self.project.time_frames['reporting'].tuple),
                'entity': model.entity,
                'dependent_variable': model.Y_col,
                'model_frequency': model.frequency,
                'project_name': project_name,
                'graph_name': graph_name,
            })
            pprint(model.report)
            if df is None:
                pd.DataFrame.from_dict(model.report, orient='index')
            else:
                pass

            now = formatted_now()
            filename = f'report_{project_name}--{graph_name}--{now}.csv'
            filepath = os.path.join(export_dir, filename)
            df.to_csv(filepath)

    def whosthere(self):
        my_tuple = self.equipment['chiller'].energy_models['TOWT'].time_frames['baseline'].tuple
        start_date = my_tuple[0]
        end_date = my_tuple[1]
        print(f'model train start date: {start_date} \nmodel train end date: {end_date}')


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
            self.time_frames = project.time_frames
        res = self.get_unique_entity_for_model(name) #todo: binds energy model to entity, correct?
        self.entity = res
        self.Y_series = None
        self.energy_models = {}

    def get_unique_entity_for_model(self, name=None, brick_class=None):
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
            print(f"Found unique entity named {name} in graph. \n")

        return res.entities_list[0]


    def feature_engineering(self):
        """

        This method applies transformations to time-series data that are unique to particular entity types. The
        transformations apply to BMS data, and the end goal is to return a good Y series on which to train an energy
        model. Relies on hard-coded brick conventions (e.g. brick class names, see ontology at BrickSchema.org)

        Theoretically, these feature engineering functions should not change building-to-building, but one-off
        customizations may be needed inevitably. Try to use conditionals, etc. to accommodate as many cases as
        possible. E.g., for heating and chiller systems, first check if there is a btu or power meter. If not,
        then ues the 'pseudo-Btu' approach. This could apply to any hot/cold water system.

        The outcome of this function is to update Y_series with the newly feature-engineering timeseries data.

        """
        name = self.name
        brick_class = self.entity.brick_class
        if brick_class == 'Hot_Water_System':
            pumps_df = None
            temps_df = None
            for entity in self.system_entities.entities_list:
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
            pseudo_Btus_series = pumps_total_df * dTemp_df
            self.Y_series = pseudo_Btus_series
            self.Y_series.name = 'pseudo_Btus'
        elif brick_class == 'Chiller':
            for key, value in self.data.items():
                if isinstance(value, TimeseriesResponse):
                    if self.data[key].brick_class == 'Electric_Power_Sensor':
                        self.Y_series = self.data[key].data.iloc[:, 0]
                        self.Y_series.name = 'chiller_power'
                        #ToDo: cleaning this up would require changing structure of timeseries responses/attributes
                        # for equipment
        else:
            msg = f'No feature engineering set up for entities of type {type}'
            raise Exception(msg)

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

    def get_data(self, time_frame='total'):
        """Get timeseries data for each system in the model.

        :param project:
        :return:
        """
        #ToDo: may, in this method, want to check if there are already specified time-series for model data,
        # rather than returning all timeseries of a system (which is fine for now_
        time_frame_ = self.project.time_frames[time_frame].tuple
        df = self.project.brick_graph.get_system_timeseries(
            self.name, time_frame_
        )
        # now, unpack populated timeseries data from self.project.brick_graph.systems[name].entities_list and write
        # them to system_entities.entities_list ... there is probably a cleaner way to have gone about this data request
        self.system_entities.entities_list = self.project.brick_graph.systems[self.name].entities_list
        self.dataframe = df

class Equipment(GraphEntity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def get_data(self, time_frame='total'):
        """Get timeseries data for each system in the model.

        :param project:
        :return:
        """
        #ToDo: may, in this method, want to check if there are already specified time-series for model data,
        # rather than returning all timeseries of a system (which is fine for now_
        time_frame_ = self.project.time_frames[time_frame].tuple
        res = self.entity.get_all_timeseries(
            time_frame_
        )
        self.data = res  #ToDo: clean this up w/r/t System.get_data()

class ModelPlus():
    '''This class exists for the sole purpose of tacking on additional attributes to an imported model from the
    energy_models subrepo. The attributes are related to the project, building, system, equipment etc that don't
    really belong in the energy_models subrepo. These additional attributes can't be just added to TOWT or TODT in
    energy_models because we don't want the subrepo to "know about" (contain any code for) this EMIS-brick repo.

    '''
    def __init__(self, instance, entity_name=None, model_name=None, **kwargs):
        '''Initialization. Copies selected attributes from the EnergyModelset instance.

        :param instance: this needs to be an instance of the Modelset class
        :param entity_name: (str) must match a name found either in the "systems" attribute or the "equipment"
        attribute of the EnergyModelset instance.
        :param model_name: (str) must match the name of a model found in the "energy_models" attribute of the specified
        entity
        :param kwargs:
        '''
        self.project = instance.project
        if entity_name is None:
            msg = 'ModelPlus instance needs an entity_name argument. This needs to be either the name of a piece of ' \
                  'equipment, or the name of a system.'
            raise Exception(msg)
        if model_name is None:
            msg = 'ModelPlus instance needs a model_name argument, which must match the name of an energy model ' \
                  'within the "energy_models" attribute of the entity (which is either a system or an equipment).'
            raise Exception(msg)
        try:
            entity = instance.systems[entity_name]
        except KeyError:
            entity = instance.equipment[entity_name]
        except KeyError:
            msg = f'No system or equipment found in {instance} with name {entity_name}.'
            raise Exception(msg)
        try:
            model = entity.energy_models[model_name]
        except KeyError:
            msg = f'In {instance}, no model found in {entity_name} named {model_name}.'
            raise Exception(msg)
        model.location = self.project.location
        self.model = model
        self.entity = entity