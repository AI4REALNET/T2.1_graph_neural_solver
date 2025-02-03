import os
import itertools
from typing import Union
import warnings
import pathlib
import math
import torch
from torch.nn import Module
from lips.config import ConfigManager
from lips.logger import CustomLogger
from lips.dataset import DataSet

from gnn_powergrid.dataset.utils.graph_utils import get_loader
from gnn_powergrid.dataset.utils.graph_utils import get_all_active_powers
from gnn_powergrid.dataset.utils.solver_utils import get_obs_with_config

from gnn_powergrid.gnn.layers import GPGinput
from gnn_powergrid.gnn.layers import GPGintermediate
from gnn_powergrid.gnn.layers import LocalConservationLayer

class GPGmodel(Module):
    def __init__(self,
                 sim_config_path: Union[pathlib.Path, str],
                 sim_config_name: Union[str, None]=None,
                 name: Union[str, None]=None,
                 log_path: Union[None, pathlib.Path, str]=None,
                 **kwargs):
        super().__init__()
        if not os.path.exists(sim_config_path):
            raise RuntimeError("Configuration path for the simulator not found!")
        if not str(sim_config_path).endswith(".ini"):
            raise RuntimeError("The configuration file should have `.ini` extension!")
        # config
        self.sim_config_name = sim_config_name if sim_config_name is not None else "DEFAULT"
        self.sim_config = ConfigManager(section_name=self.sim_config_name, path=sim_config_path)
        self.name = name if name is not None else self.sim_config.get_option("name")
        self.default_dtype = torch.float32 if "default_dtype" not in kwargs else kwargs.get("default_dtype")
        # logging
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, log_path).logger
        # model parameters
        self.params = self.sim_config.get_options_dict()
        self.params.update(kwargs)
        # sizes
        self.input_size = None if kwargs.get("input_size") is None else kwargs["input_size"]
        self.output_size = None if kwargs.get("output_size") is None else kwargs["output_size"]
        
    def build_model(self, **kwargs):
        self.input_layer = GPGinput(ref_node=self.params["ref_node"],
                                    latent_dimension=self.params["latent_dimension"],
                                    hidden_layers=self.params["hidden_layers"],
                                    input_dim=self.input_size,
                                    output_dim=self.output_size,
                                    device=self.params["device"]
                                    )
        self.lc_layer = LocalConservationLayer()
        self.inter_layers = torch.nn.ModuleList([GPGintermediate(ref_node=self.params["ref_node"], 
                                                                 device=self.params["device"])
                                                 for _ in range(self.params["num_gnn_layers"])])
        
    def forward(self, batch):
        errors = []
        out, _ = self.input_layer(batch)
        nodal_error = self.lc_layer(batch, out)
        errors.append(abs(nodal_error).sum())
        
        for layer in self.inter_layers:
            out, _ = layer(batch, out)
            nodal_error = self.lc_layer(batch, out)
            errors.append(abs(nodal_error).sum())
            
        return out, errors
    
    def _do_forward(self, batch):
        self._batch = batch
        predictions, errors = self.forward(batch)
        return predictions, errors
    
    def process_dataset(self, dataset: DataSet, training: bool=True, **kwargs):
        """
        Could do some preprocessing stuff like normalization if required 
        before passing the values to neural network
        """        
        if training:
            batch_size = self.params["train_batch_size"]
        else:
            batch_size = self.params["eval_batch_size"]
        
        warnings.filterwarnings("ignore")
        _, obs = get_obs_with_config(self.params)
        data_loader = get_loader(obs, dataset.data, batch_size, self.params["device"])
        return data_loader
    
    def _post_process(self, data):
        """post processing on a given data
        
        Could transform the voltage angles to powers for example
        
        Here we transform the voltage angles predicted by GNN in Radians to Degrees which are
        required for the computation of active powers (in recontruct_output)

        Parameters
        ----------
        data : _type_
            data here is voltage angels (theta) in radians which are predicted by the GNN
        """
        # transform the radians to degrees
        theta_degree = data * (180/math.pi)
        return theta_degree
    
    def _infer_size(self, data):
        self.input_size = None
        self.output_size = None
    
    def _reconstruct_output(self, data, dataset):
        """
        Reconstruct the output of GNN to be compatible with Evaluation format
        TODO: to put the transformed powers in dictionary for example
        
        Parameters
        ----------
        data : _type_
            data here is thetas in degrees which are predicted by GNN and post-processed
        """
        outputs = {}
        _, obs = get_obs_with_config(self.params)
        outputs["p_or"], outputs["p_ex"] = get_all_active_powers(dataset,
                                                                 obs,
                                                                 theta_bus=data.cpu().view(-1, obs.n_sub*2))
        
        return outputs
    
    def get_metadata(self):
        res_json = {}
        res_json["input_size"] = self.input_size
        res_json["output_size"] = self.output_size
        
    
    
