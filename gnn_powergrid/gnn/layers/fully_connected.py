from torch import nn

class FullyConnected(nn.Module):
    """Fully Connected (dense) layer

    Parameters
    ----------
    latent_dimension : int, optional
        _description_, by default 10
    hidden_layers : int, optional
        _description_, by default 3
    input_dim : _type_, optional
        _description_, by default None
    output_dim : _type_, optional
        _description_, by default None
    """
    def __init__(self, latent_dimension=10, hidden_layers=3, input_dim=None, output_dim=None):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.latent_dimension = latent_dimension
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.build_model()

    def build_model(self):
        """construct the architecture elements
        """
        for layer in range(self.hidden_layers):
            left_dim = self.latent_dimension
            right_dim = self.latent_dimension
            if (layer == 0) and (self.input_dim is not None):
                left_dim = self.input_dim
            if (layer == self.hidden_layers-1) and (self.output_dim is not None):
                right_dim = self.output_dim
            self.layers.append(nn.Linear(in_features=left_dim, out_features=right_dim))

    def forward(self, h):
        """forward pass of the FC layer

        Parameters
        ----------
        h : ``torch.Tensor`` or ``numpy.Array``
            inputs to the forward pass

        Returns
        -------
        _type_
            _description_
        """
        for layer in range(self.hidden_layers):
            if layer == self.hidden_layers-1:
                h = self.layers[layer](h)
            else:
                h = nn.functional.leaky_relu(self.layers[layer](h))
        return h