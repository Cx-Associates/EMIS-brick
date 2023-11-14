"""

"""

from subrepos.energy_models.src.utils import Model, TOWT

class Project():
    """

    """
    def __init__(self, **kwargs):
        self.location = None
        self.__dict__.update(kwargs.copy())
        if isinstance(self.location, str):
            pass #ToDo: add code to resolve lat, long as a function of place, e.g. google geocode REST api
        if isinstance(self.time_frame, str):
            if self.time_frame == 'past week':
                self.time_frame_formatted = None #ToDo: auto-parse the project input arguments to api-friendly


class EnergyModelset():
    """

    """
    def __init__(self, project, models):
        self.project = project
        self.models = models

        for model_name in self.models:
            instance = Model()
            self.__setattr__(model_name, instance)