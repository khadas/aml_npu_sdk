from acuitylib.layer.customlayer import CustomLayer
from acuitylib.layer.acuitylayer import IoMap

class SyncHost(CustomLayer):

    op = 'sync_host'

    # label, description
    def_input  = [IoMap('in0', 'in', 'input port')]
    def_output = [IoMap('out0', 'out', 'output port')]

    def setup(self, inputs, outputs):
        outputs[0].shape = inputs[0].shape.copy()

