from .base import BaseModel

from Markov_Models.models.base import BaseMicroMSM, BaseMacroMSM
class MSM(BaseModel):
    def __init__(self, *args, **kwargs):
        BaseModel.__init__(self, *args, **kwargs)
        self.microstates = BaseMicroMSM(self)
        self.macrostates = BaseMacroMSM(self)

class HMM(BaseModel):
    def __init__(self, *args, **kwargs):
        BaseModel.__init__(self, *args, **kwargs)

class Markov_Chain(BaseModel):
    def __init__(self, *args, **kwargs):
        BaseModel.__init__(self, *args, **kwargs)
