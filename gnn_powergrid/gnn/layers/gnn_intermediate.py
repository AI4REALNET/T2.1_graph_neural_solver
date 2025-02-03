import torch
import numpy as np
from torch_geometric.nn import MessagePassing

class GPGintermediate(MessagePassing):
    """Graph Power Grid intermediate layer

    This is the intermediate layer of GNN that gets the theta from the previous layer and
    updates them through power flow equation
    """
    def __init__(self,
                 ref_node,
                 device="cpu"):
        super().__init__(aggr="add")
        self.ref_node = ref_node
        self.theta = None
        self.device = device
        
    def forward(self, batch, theta):
        """The forward pass of message passing network

        Args:
            batch (_type_): the data batch
            theta (_type_): the voltage angle from the previous layer

        Returns:
            _type_: the updated voltage angles
        """
        self.theta = theta
        
        # Compute a message and propagate it to each node, it does 3 steps
        # 1) It computes a message (Look at the message function below)
        # 2) It propagates the message using an aggregation (sum here)
        # 3) It calls the update function which could be Neural Network
        aggr_msg = self.propagate(batch.edge_index_no_diag,
                                  y=self.theta,
                                  edge_weights=batch.edge_attr_no_diag * 100.0
                                 )

        n_bus = batch.ybus.size()[1]
        # keep only the diagonal elements of the ybus 3D tensors for denominator part
        ybus = batch.ybus.view(-1, n_bus, n_bus) * 100.0
        # ybus = ybus * torch.eye(*ybus.shape[-2:], device=self.device).repeat(ybus.shape[0], 1, 1)
        # denominator = ybus[ybus.nonzero(as_tuple=True)].view(-1,1)
        denominator = torch.hstack([ybus[i].diag() for i in range(len(ybus))]).reshape(-1,1)
        
        input_node_power = (batch.x[:,0] - batch.x[:,1]).view(-1,1)
        numerator = input_node_power - aggr_msg
        # out = (input_node_power - aggr_msg) / denominator
        indices = torch.where(denominator.flatten()!=0.)[0]
        out = torch.zeros_like(denominator)
        out[indices] = torch.divide(numerator[indices], denominator[indices])

        #we impose that reference node has theta=0
        out = out.view(-1, n_bus, 1) - out.view(-1,n_bus,1)[:,self.ref_node].repeat_interleave(n_bus, 1).view(-1, n_bus, 1)
        out = out.flatten().view(-1,1)
        #we impose the not used buses to have theta=0
        out[denominator==0] = 0
        
        return out, aggr_msg

    def message(self, y_i, y_j, edge_weights):
        tmp = y_j * edge_weights.view(-1,1)
        return tmp

    def update(self, aggr_out):
        return aggr_out