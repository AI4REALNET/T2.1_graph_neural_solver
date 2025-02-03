import warnings
import copy
import math
from tqdm.auto import tqdm
import numpy as np
from scipy.sparse import csr_matrix
import torch
import grid2op
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from grid2op.Parameters import Parameters
from lightsim2grid.lightSimBackend import LightSimBackend

def create_fake_obs(obs, data, idx = 0):
    """Create a fake observation from the env by copying data values

    Args:
        obs (_type_): _description_
        data (_type_): _description_
        idx (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    obs.line_status = data["line_status"][idx]
    obs.topo_vect = data["topo_vect"][idx]
    return obs



def get_features_per_bus(obs, dataset, index: int = 0):
    n_buses = obs.n_sub * 2
    feature_matrix = np.zeros((n_buses, 2), dtype=np.float32)
    for sub_ in range(obs.n_sub):
        objects = obs.get_obj_connect_to(substation_id=sub_)

        if len(objects["generators_id"]) > 0:
            for gen_ in objects["generators_id"]:
                if obs.gen_bus[gen_] == 1:
                    feature_matrix[sub_, 0] += np.sum(dataset.get("prod_p")[index, gen_])
                if obs.gen_bus[gen_] == 2:
                    feature_matrix[sub_ + obs.n_sub, 0] += np.sum(dataset.get("prod_p")[index, gen_])
        if len(objects["loads_id"]) > 0:
            for load_ in objects["loads_id"]:
                if obs.load_bus[load_] == 1:
                    feature_matrix[sub_,1] += np.sum(dataset.get("load_p")[index, load_])
                if obs.load_bus[load_] == 2:
                    feature_matrix[sub_+obs.n_sub,1] += np.sum(dataset.get("load_p")[index, load_])
        
    return feature_matrix

def get_all_features_per_bus(obs, dataset):
    features = torch.zeros((len(dataset["prod_p"]), obs.n_sub*2, 2))
    for i in range(len(features)):
        obs = create_fake_obs(obs, dataset, idx=i)
        features[i, :, :] = torch.tensor(get_features_per_bus(obs, dataset, index=i))
    return features.float()

def get_theta_node(obs, sub_id, bus):
    """Get the voltage angles for a specific substation

    Args:
        obs (_type_): Grid2op observation
        sub_id (int): the idenitifer of a substation
        bus (int): the bus number

    Returns:
        float: returns the voltage angle at the node
    """
    obj_to_sub = obs.get_obj_connect_to(substation_id=sub_id)

    lines_or_to_sub_bus = [i for i in obj_to_sub['lines_or_id'] if obs.line_or_bus[i] == bus]
    lines_ex_to_sub_bus = [i for i in obj_to_sub['lines_ex_id'] if obs.line_ex_bus[i] == bus]

    thetas_node = np.append(obs.theta_or[lines_or_to_sub_bus], obs.theta_ex[lines_ex_to_sub_bus])
    thetas_node = thetas_node[thetas_node != 0]

    theta_node = 0.
    if len(thetas_node) != 0:
        theta_node = np.max(thetas_node)

    return theta_node

def get_theta_bus_topo(dataset, obs):
    
    Ybus = dataset["YBus"]
    bus_theta = np.zeros((Ybus.shape[0], obs.n_sub*2), dtype=complex)
    
    for idx in range(Ybus.shape[0]):
        #obs.topo_vect = dataset["topo_vect"][idx]
        obs = create_fake_obs(obs, dataset, idx)
        obs.theta_or = dataset["theta_or"][idx]
        obs.theta_ex = dataset["theta_ex"][idx]
        for sub_ in range(obs.n_sub):
            bus_theta[idx, sub_] = get_theta_node(obs, sub_id=sub_, bus=1)
            bus_theta[idx, sub_+obs.n_sub] = get_theta_node(obs, sub_id=sub_, bus=2)

    return bus_theta

def get_target_variables_per_bus(obs, dataset):
    targets = torch.tensor(get_theta_bus_topo(dataset, obs).real).unsqueeze(dim=2)
    return targets.float()

def get_batches_pyg(obs,
                    edge_indices,
                    edge_indices_no_diag,
                    features,
                    targets,
                    ybuses,
                    batch_size=128,
                    device="cpu",
                    edge_weights=None,
                    edge_weights_no_diag=None):
    """Create Pytorch Geometric based data loaders
    This function gets the features and targets at nodes and create batches of structured data

    Args:
        edge_indices (list): list of edge indices 
        edge_indices_no_diag (_type_): list of edge indices without the self loops (without diagonal elements of adjacency matrix)
        features (_type_): list of features (injections)
        targets (_type_): list of targets (theta)
        ybuses (_type_): admittance matrix
        batch_size (int, optional): _description_. Defaults to 128.
        device (str, optional): _description_. Defaults to "cpu".
        edge_weights (_type_, optional): edge weight which is the admittance matrix element between two nodes. Defaults to None.
        edge_weights_no_diag (_type_, optional): edge weight without values for diagonal elements of adjacency matrix. Defaults to None.

    Returns:
        _type_: _description_
    """
    torchDataset = []
    for i, feature in enumerate(features):
        if edge_weights is not None:
            edge_weight = torch.tensor(edge_weights[i], dtype=feature.dtype)
            edge_weight_no_diag = torch.tensor(edge_weights_no_diag[i], dtype=feature.dtype)
        else:
            edge_weight = None
            edge_weight_no_diag = None
            
        if isinstance(ybuses, csr_matrix):
            ybus = np.squeeze(np.asarray(ybuses[i].todense()))
            ybus = ybus.reshape(obs.n_sub*2, obs.n_sub*2)
        else:
            ybus = ybuses[i]
            
        sample_data = Data(x=feature,
                           y=targets[i],
                           edge_index=torch.tensor(edge_indices[i]),
                           edge_index_no_diag=torch.tensor(edge_indices_no_diag[i]),
                           edge_attr=edge_weight,
                           edge_attr_no_diag=edge_weight_no_diag,
                           ybus=torch.tensor(ybus.real, dtype=torch.float32))
        sample_data.to(device)
        torchDataset.append(sample_data)
    loader = DataLoader(torchDataset, batch_size=batch_size)

    return loader

def get_edge_index_from_ybus(ybus_matrix, obs, add_loops=True) -> list:
    """Get all the edge_indices from Ybus matrix

    Parameters
    ----------
    ybus_matrix : _type_
        Ybus matrix as input (NxMxM)
        with N number of observations
        and M number of nodes in the graph

    Returns
    -------
    ``list``
        a list of edge indices
    """
    # ybus_mat = copy.deepcopy(ybus_matrix)
    edge_indices = []
    #for ybus in ybus_mat:
    for ybus in ybus_matrix:
        ybus_cpy = copy.deepcopy(ybus)
        if isinstance(ybus_cpy, csr_matrix):
            ybus_cpy = np.squeeze(np.asarray(ybus_cpy.todense()))
            ybus_cpy = ybus_cpy.reshape(obs.n_sub*2, obs.n_sub*2)
        if not(add_loops):
            np.fill_diagonal(ybus_cpy, val=0.)
        bus_or, bus_ex = np.where(ybus_cpy)
        edge_index = np.column_stack((bus_or, bus_ex)).T
        edge_indices.append(edge_index)
    return edge_indices

def get_edge_weights_from_ybus(ybus_matrix, edge_indices, obs) -> list:
    """Get edge weights corresponding to each edge index

    Parameters
    ----------
    ybus_matrix : _type_
        _description_
    edge_indices : _type_
        edge indices returned by the get_edge_index_from_ybus function

    Returns
    -------
    ``list``
        a list of edge weights
    """
    edge_weights = []
    for edge_index, ybus in zip(edge_indices, ybus_matrix):
        edge_weight = []
        if isinstance(ybus, csr_matrix):
            ybus = np.squeeze(np.asarray(ybus.todense()))
            ybus = ybus.reshape(obs.n_sub*2, obs.n_sub*2)
        for i in range(edge_index.shape[1]):
            edge_weight.append(ybus[edge_index[0][i], edge_index[1][i]])
        edge_weight = np.array(edge_weight)
        edge_weights.append(edge_weight)
    return edge_weights

def get_loader(obs, data, batch_size, device):
    """
    This function structures the features, targets, edge_indices and edge weights through a GNN
    point of view and create a data loader for a given dataset.

    Args:
        obs (_type_): an observation of environment
        data (dict): the dataset
        batch_size (int): the batch size considered for data loader
        device (str): the device on which the computation should be performed

    Returns:
        _type_: _description_
    """
    pbar = tqdm(range(4))
    pbar.set_description("Get Features")
    features = get_all_features_per_bus(obs, data)
    pbar.update(1)
    pbar.set_description("Get Targets")
    targets = get_target_variables_per_bus(obs, data)
    pbar.update(1)
    pbar.set_description("Get edge_index info")
    edge_indices = get_edge_index_from_ybus(data["YBus"], obs, add_loops=True)
    edge_weights = get_edge_weights_from_ybus(data["YBus"], edge_indices, obs)
    edge_indices_no_diag = get_edge_index_from_ybus(data["YBus"], obs, add_loops=False)
    edge_weights_no_diag = get_edge_weights_from_ybus(data["YBus"], edge_indices_no_diag, obs)
    pbar.update(1)
    pbar.set_description("Create loader")
    loader = get_batches_pyg(obs,
                             edge_indices=edge_indices,
                             edge_indices_no_diag=edge_indices_no_diag,
                             features=features,
                             targets=targets,
                             ybuses=data["YBus"],
                             edge_weights=edge_weights,
                             edge_weights_no_diag=edge_weights_no_diag,
                             batch_size=batch_size,
                             device=device)
    pbar.update(1)
    pbar.close()

    return loader

def prepare_dataset(benchmark, batch_size=128, device="cpu"):
    """Prepare the dataset for GNN based model

    Args:
        benchmark (_type_): _description_
        batch_size (int, optional): _description_. Defaults to 128.
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        _type_: _description_
    """
    warnings.filterwarnings("ignore")
    params = Parameters()
    params.ENV_DC = True
    env = grid2op.make(benchmark.env_name, param=params, backend=LightSimBackend())
    obs = env.reset()
    
    # Train dataset
    print("*******Train dataset*******")
    print(f"Train data size: {benchmark.train_dataset.size}")
    train_loader = get_loader(obs, benchmark.train_dataset.data, batch_size, device)
    # Val dataset
    print("*******Validation dataset*******")
    print(f"Validation data size: {benchmark.val_dataset.size}")
    val_loader = get_loader(obs, benchmark.val_dataset.data, batch_size, device)
    
    # Test dataset
    print("*******Test dataset*******")
    print(f"Test data size : {benchmark._test_dataset.size}")
    test_loader = get_loader(obs, benchmark._test_dataset.data, batch_size, device)

    # OOD dataset
    print("*******OOD dataset*******")
    print(f"OOD data size: {benchmark._test_ood_topo_dataset.size}")
    test_ood_loader = get_loader(obs, benchmark._test_ood_topo_dataset.data, batch_size, device)

    return train_loader, val_loader, test_loader, test_ood_loader

def get_active_power(dataset, obs, theta, index):
    """
    Computes the active power (flows) from thetas (subs) for a specific index

    Parameters
    ----------
    dataset : _type_
        data
    obs : _type_
        Grid2op observation
    theta : _type_
        voltage angle
    index : _type_
        the observation index for which the active powers should be computed  

    Returns
    -------
    _type_
        _description_
    """
    lor_bus, lor_conn = obs._get_bus_id(obs.line_or_pos_topo_vect, obs.line_or_to_subid)
    lex_bus, lex_conn = obs._get_bus_id(obs.line_ex_pos_topo_vect, obs.line_ex_to_subid)
    index_array = np.vstack((np.arange(obs.n_line), lor_bus, lex_bus)).T 
    # Create the adjacency matrix (MxN) M: branches and N: Nodes
    A_or = np.zeros((obs.n_line, obs.n_sub*2))
    A_ex = np.zeros((obs.n_line, obs.n_sub*2))

    for line in index_array[:,0]:
        if index_array[line,1] != -1:
            A_or[line, index_array[line,1]] = 1
            A_or[line, index_array[line,2]] = -1
            A_ex[line, index_array[line,1]] = -1
            A_ex[line, index_array[line,2]] = 1
    
    # Create the diagonal matrix D (MxM)
    Ybus = dataset["YBus"][index]
    if isinstance(dataset["YBus"], csr_matrix):
        Ybus = np.squeeze(np.asarray(Ybus.todense()))
        Ybus = Ybus.reshape(obs.n_sub*2, obs.n_sub*2)
    D = np.zeros((obs.n_line, obs.n_line), dtype=complex)
    for line in index_array[:, 0]:
        bus_from = index_array[line, 1]
        bus_to = index_array[line, 2]
        D[line,line] = Ybus[bus_from, bus_to] * (-1)

    # Create the theta vector ((M-1)x1)
    theta = 1j*((theta[index,1:]*math.pi)/180)
    p_or = (D.dot(A_or[:,1:])).dot(theta.reshape(-1,1))
    p_ex = (D.dot(A_ex[:,1:])).dot(theta.reshape(-1,1))

    #return p_or, p_ex
    return p_or.imag * 100 , p_ex.imag * 100

def get_all_active_powers(dataset, obs, theta_bus):
    """Computes all the active powers 
    
    It computes the active powers for all the observations from thetas (voltage angles) at bus

    Parameters
    ----------
    dataset : _type_
        the data
    obs : _type_
        Grid2op observation
    theta_bus : _type_
        the voltage angles at buses

    Returns
    -------
    _type_
        numpy arrays corresponding to active powers at the origin and extremity side of power lines
    """
    data_size = len(dataset["p_or"])
    p_or = np.zeros_like(dataset["p_or"])
    p_ex = np.zeros_like(dataset["p_ex"])

    #theta_bus = get_theta_bus(dataset, obs)
    for ind in tqdm(range(data_size)):
        obs = create_fake_obs(obs, dataset, ind)
        p_or_computed, p_ex_computed = get_active_power(dataset, obs, theta_bus, index=ind)
        p_or[ind, :] = p_or_computed.flatten()
        p_ex[ind, :] = p_ex_computed.flatten()
    
    return p_or, p_ex

def get_active_powers_batch(dataset, obs, theta_bus):
    """Computes all the active powers 
    
    It computes the active powers for all the observations from thetas (voltage angles) at bus

    Parameters
    ----------
    dataset : _type_
        the data
    obs : _type_
        Grid2op observation
    theta_bus : _type_
        the voltage angles at buses

    Returns
    -------
    _type_
        numpy arrays corresponding to active powers at the origin and extremity side of power lines
    """
    #data_size = len(dataset["p_or"])
    data_size = theta_bus.size()[0]
    p_or = np.zeros((data_size, dataset["p_or"].shape[1]))
    p_ex = np.zeros((data_size, dataset["p_or"].shape[1]))

    #theta_bus = get_theta_bus(dataset, obs)
    for ind in tqdm(range(data_size)):
        obs = create_fake_obs(obs, dataset, ind)
        p_or_computed, p_ex_computed = get_active_power(dataset, obs, theta_bus, index=ind)
        p_or[ind, :] = p_or_computed.flatten()
        p_ex[ind, :] = p_ex_computed.flatten()
    
    return p_or, p_ex
