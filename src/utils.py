"""

"""

from subrepos.energy_models.src.utils import Model, TOWT
from subrepos.brickwork.utils import BrickModel

class Project():
    """

    """
    def __init__(self, **kwargs):
        self.location = None
        self.brick_model = None
        self.__dict__.update(kwargs.copy())
        if isinstance(self.location, str):
            pass #ToDo: add code to resolve lat, long as a function of place, e.g. google geocode REST api
        if isinstance(self.time_frame, str):
            if self.time_frame == 'past week':
                self.time_frame_formatted = None #ToDo: auto-parse the project input arguments to api-friendly

    def set_metadata(self, model_path):
        self.brick_model = BrickModel(model_path)

class EnergyModelset():
    """

    """
    def __init__(self, project, models):
        self.project = project
        self.models = models
        self.brick_model = project.brick_model
        for model_name in self.models:
            instance = EnergyModel(model_name, self.brick_model)
            self.__setattr__(model_name, instance)


class EnergyModel(Model):
    """

    """
    def __init__(self, name, brick_model=None):
        super().__init__()
        self.brick_model = brick_model
        self.get_equipment(name)

    def get_equipment(self, name=None, class_=None):
        brick_model = self.brick_model
        res_list = brick_model.get_entities(name, class_)

    def train(self, predict, functionOf):
        """

        :param predict:
        :param functionOf:
        :return:
        """
        if functionOf == 'TOWT':
            pass