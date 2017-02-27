import Markov_Models as mm
from Markov_Models.models.base import BaseModel, BaseMicroMSM, BaseMacroMSM


class MSM(BaseModel):
    def __init__(self, *args, **kwargs):
        BaseModel.__init__(self, *args, **kwargs)
        self.microstates = BaseMicroMSM(self)
        self.macrostates = BaseMacroMSM(self)

class HMM(BaseModel):
    pass

class Markov_Chain(BaseModel):
    pass
