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

    def set_metadata(self, model_path):
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
            pass
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
    def __init__(self, project, models):
        self.project = project
        self.model_names = models
        self.brick_model = project.brick_model
        for model_name in self.model_names:
            instance = EnergyModel(
                model_name,
                project
            )
            self.__setattr__(model_name, instance)

    def get_data(self):
        for energy_model_name in self.model_names:
            model = getattr(self, energy_model_name)
            model.get_data(self.project)

class EnergyModel(Model):
    """

    """
    def __init__(self, name, project):
        super().__init__()
        self.name = name
        self.project = project
        self.get_equipment(name)

    def get_equipment(self, name=None, class_=None):
        brick_model = self.project.brick_model
        res_list = brick_model.get_entities(name, class_)

    def get_data(self, project):
        """

        :param project:
        :return:
        """
        brick_model = project.brick_model.get_data(
            self.name, project
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