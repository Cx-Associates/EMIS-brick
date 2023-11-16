"""

"""

from subrepos.energy_models.src.utils import Model, TOWT
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
    """

    """
    def __init__(self, **kwargs):
        self.location = None
        self.brick_model = None
        self.__dict__.update(kwargs.copy())
        if isinstance(self.location, str):
            pass #ToDo: add code to resolve lat, long as a function of place, e.g. google geocode REST api

    def load_graph(self, model_path):
        """

        :param model_path:
        :return:
        """
        self.brick_model = BrickModel(model_path)

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

class EnergyModelset():
    """

    """
    def __init__(self, project, systems):
        self.project = project
        self.systems = {}
        self.brick_model = project.brick_model
        for system_name in systems:
            instance = EnergyModel(
                system_name,
                project
            )
            self.systems.update({system_name: instance})

    def get_data(self):
        for k, v in self.systems.items():
            v.get_data(self.project)

class EnergyModel(Model):
    """

    """
    def __init__(self, name, project):
        super().__init__()
        self.name = name
        self.project = project
        self.check_system(name)

    def check_system(self, name=None):
        brick_model = self.project.brick_model
        res_list = brick_model.get_entities(name)
        if len(res_list) == 0:
            raise Exception(f"Couldn't find an entity named {name} in the loaded graph.")
        elif len(res_list) > 1:
            raise Exception(f"Found more than one entity named {name} in the loaded graph. Each system in the "
                            f"modelset must have a unique name; otherwise you should either modify the graph (make a "
                            f"new, abritrary system) or modify this function to filter the graph query based on more "
                            f"than simply the name of a system.")
        else:
            print(f"Found entity named {name} in graph.")

    def get_data(self, project):
        """

        :param project:
        :return:
        """
        time_frame = project.time_frames['total'].tuple
        project.brick_model.get_system_timeseries(
            self.name, time_frame
        )
        pass

    def train(self, predict, functionOf):
        """

        :param predict:
        :param functionOf:
        :return:
        """
        if 'TOWT' in functionOf:
            train_start = self.project.time_frames['baseline'].tuple[0]
            train_end = self.project.time_frames['baseline'].tuple[1]
            towt = TOWT(
                self,
                train_start=train_start,
                train_end=train_end
            )
            pass