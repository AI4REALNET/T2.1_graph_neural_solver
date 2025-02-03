from torch_geometric.nn import MessagePassing

class LocalConservationLayer(MessagePassing):
    """Compute local conservation error

    This class computes the local conservation error without any update of voltage angles.

    Args:
        MessagePassing (_type_): _description_
    """
    def __init__(self):
        super().__init__(aggr="add")
        self.thetas = None
        
    def forward(self, batch, thetas=None):
        # theta from previous GNN layer
        self.thetas = thetas

        # The difference with GPG layers resides also in propagation which gets the edge_index
        # with self loops (with diagonal elements of adjacency matrix)
        aggr_message = self.propagate(batch.edge_index,
                                      y=self.thetas,
                                      edge_weights=batch.edge_attr * 100)

        input_node_power = (batch.x[:,0] - batch.x[:,1]).view(-1,1)
        # compute the local conservation error (at node level)
        nodal_error = input_node_power - aggr_message

        return nodal_error

    def message(self, y_i, y_j, edge_weights):
        """
        Compute the message
        """
        tmp = y_j * edge_weights.view(-1,1)
        return tmp
    