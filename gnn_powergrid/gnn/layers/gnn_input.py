import torch
from torch_geometric.nn import MessagePassing
from gnn_powergrid.gnn.layers import FullyConnected


class GPGinput(MessagePassing):
    """Graph Power Grid Input layer

    This is the input layer of GNN initialize the theta (voltage angles) using a MLP and
    updates them through power flow optimization as a loss during training

    """
    def __init__(self,
                 ref_node,
                 latent_dimension=20,
                 hidden_layers=3,
                 input_dim=2,
                 output_dim=1,
                 device="cpu"
                 ):
        super().__init__(aggr="add")
        self.ref_node = ref_node
        self.theta = None
        self.device = device
        self.latent_dimension = latent_dimension
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        # when hidden_layers = 1 it is equivalent to a Linear layer
        self.fc = FullyConnected(latent_dimension=self.latent_dimension,
                                 hidden_layers=self.hidden_layers,
                                 input_dim=self.input_dim,
                                 output_dim=self.output_dim)
        self.fc.to(device)

    def forward(self, batch):

        # self.theta = torch.zeros_like(batch.y, dtype=batch.y.dtype)
        # self.theta = self.linear(batch.x)
        self.theta = self.fc(batch.x)
        # self.theta = self.theta.view(-1,14,1) - self.theta.view(-1,14,1)[:,0].repeat_interleave(14,1).view(-1,14,1)
        # self.theta = self.theta.flatten().view(-1,1)

        aggr_msg = self.propagate(batch.edge_index_no_diag,
                                  y=self.theta,
                                  edge_weights=batch.edge_attr_no_diag * 100.0
                                 )
        
        # keep only the diagonal elements of the ybus 3D tensors
        n_bus = batch.ybus.size()[1]
        n_sub = n_bus / 2
        ybus = batch.ybus.view(-1, n_bus, n_bus) * 100.0
        ybus = ybus * torch.eye(*ybus.shape[-2:], device=self.device).repeat(ybus.shape[0], 1, 1)
        denominator = torch.hstack([ybus[i].diag() for i in range(len(ybus))]).reshape(-1,1)
        
        input_node_power = (batch.x[:,0] - batch.x[:,1]).view(-1,1)
        numerator = input_node_power - aggr_msg
        
        indices = torch.where(denominator.flatten()!=0.)[0]
        out = torch.zeros_like(denominator)
        out[indices] = torch.divide(numerator[indices], denominator[indices])

        #we impose that node 0 has theta=0
        out = out.view(-1, n_bus, 1) - out.view(-1,n_bus,1)[:,self.ref_node].repeat_interleave(n_bus, 1).view(-1, n_bus, 1)
        out = out.flatten().view(-1,1)
        
        # imposing theta=0 for the bus which are not used
        out[denominator==0] = 0
        
        return out, aggr_msg
    
    def message(self, y_j, edge_weights):
        """Compute the message that should be propagated
        
        This function compute the message (which is the multiplication of theta and 
        admittance matrix elements connecting node i to j)

        Args:
            y_j (_type_): the theta (voltage angle) value at a neighboring node j
            edge_weights (_type_): corresponding edge_weight (admittance matrix element)

        Returns:
            _type_: active powers for each neighboring node
        """
        tmp = y_j * edge_weights.view(-1,1)
        return tmp
    
    def update(self, aggr_out):
        """update function of message passing layers

        We output directly the aggreated message (sum)

        Args:
            aggr_out (_type_): the aggregated message

        Returns:
            _type_: the aggregated message
        """
        return aggr_out
    