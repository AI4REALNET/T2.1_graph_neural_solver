import itertools
import torch
from gnn_powergrid.gnn.layers.gnn_input_wo_nn import GPGinput_without_NN
from gnn_powergrid.gnn.layers.local_conservation import LocalConservationLayer
from gnn_powergrid.gnn.layers.gnn_intermediate import GPGintermediate

class GPGmodel_without_NN(torch.nn.Module):
    """Create a Graph Power Grid (GPG) model without learning
    """
    def __init__(self,
                 ref_node,
                 num_gnn_layers=10,
                 device="cpu"):
        super().__init__()
        self.ref_node = ref_node
        self.num_gnn_layers = num_gnn_layers
        self.device = device

        self.input_layer = None
        self.lc_layer = None
        self.inter_layers = None

        self.build_model()

    def build_model(self):
        """Build the GNN message passing model

        It composed of a first input layer and a number of intermediate message passing layers
        These layes interleave with local conservation layers which allow to compute the error
        at the layer level
        """
        self.input_layer = GPGinput_without_NN(ref_node=self.ref_node, device=self.device)
        self.lc_layer = LocalConservationLayer()
        self.inter_layers = torch.nn.ModuleList([GPGintermediate(ref_node=self.ref_node, 
                                                                 device=self.device) 
                                                 for _ in range(self.num_gnn_layers)])

    def forward(self, batch):
        errors = []
        out, _ = self.input_layer(batch)
        nodal_error = self.lc_layer(batch, out)
        errors.append(abs(nodal_error).sum())
        
        for gnn_layer, lc_layer_ in zip(self.inter_layers, itertools.repeat(self.lc_layer)):
            out, _ = gnn_layer(batch, out)
            nodal_error = lc_layer_(batch, out)
            errors.append(abs(nodal_error).sum())

        return out, errors
