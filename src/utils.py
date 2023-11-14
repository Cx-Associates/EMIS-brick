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
        for model_name in self.models:
            instance = EnergyModel(model_name, self)
            self.__setattr__(model_name, instance)

class EnergyModel(Model):
    """

    """
    def __init__(self, name, parent):
        super().__init__()
        self.energy_modelset = parent
        self.check_name(name)

    def check_name(self, name):
        project = self.energy_modelset.project
        brick_model = project.brick_model
        res_list = brick_model.get_entities(name)
