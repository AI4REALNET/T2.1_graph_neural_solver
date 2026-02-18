import json
import pathlib
import torch
import numpy as np
from lips.benchmark.powergridBenchmark import PowerGridBenchmark

from gnn_powergrid.dataset import prepare_dataset
from gnn_powergrid.gnn.models.gnn_wo_nn import GPGmodel_without_NN
from gnn_powergrid.evaluation.evaluate_gnn import evaluate_gnn
from gnn_powergrid.utils import NpEncoder

env_name = "l2rpn_case14_sandbox"

path = pathlib.Path().resolve()
BENCH_CONFIG_PATH = path / "configs" / (env_name + ".ini")
DATA_PATH = path / "Datasets" / env_name / "DC"
LOG_PATH = path / "logs.log"

if not DATA_PATH.exists():
    DATA_PATH.mkdir(parents=True)
    

NB_SAMPLE_TRAIN = 1e2
NB_SAMPLE_VAL = 1e2
NB_SAMPLE_TEST = 1e2
NB_SAMPLE_OOD = 1e2

EPOCHS = 2

if __name__ == "__main__":

    benchmark = PowerGridBenchmark(benchmark_path=DATA_PATH,
                                   benchmark_name="Benchmark4",
                                   load_data_set=False,
                                   config_path=BENCH_CONFIG_PATH,
                                   log_path=LOG_PATH)


    benchmark.generate(nb_sample_train=int(NB_SAMPLE_TRAIN),
                    nb_sample_val=int(NB_SAMPLE_VAL),
                    nb_sample_test=int(NB_SAMPLE_TEST),
                    nb_sample_test_ood_topo=int(NB_SAMPLE_OOD),
                    do_store_physics=True,
                    is_dc=True
                    )
    
    device = torch.device("cpu") # or "cuda:0" if you have any GPU
    train_loader, val_loader, test_loader, test_ood_loader = prepare_dataset(benchmark=benchmark, 
                                                                            batch_size=128, 
                                                                            device=device)


    # use this if you have already generated some data to import them and comment two previous statements
    # benchmark = PowerGridBenchmark(benchmark_path=DATA_PATH,
    #                             benchmark_name="Benchmark4",#"DoNothing",
    #                             load_data_set=True,
    #                             config_path=BENCH_CONFIG_PATH,
    #                             log_path=LOG_PATH)

    gpg_model_wo_nn = GPGmodel_without_NN(ref_node=0, num_gnn_layers=100, device=device)
    gpg_model_wo_nn.to(device)
    
    predictions_list = []
    observations_list = []
    error_per_batch = []
    for batch in test_loader:
        out, errors = gpg_model_wo_nn(batch)
        predictions_list.append(out)
        observations_list.append(batch.y)
        #error_per_batch.append(torch.mean(torch.vstack(errors)))
        error_per_batch.append([float(error.detach().cpu().numpy()) for error in errors])
    observations = torch.vstack(observations_list)
    predictions = torch.vstack(predictions_list)
    #errors = np.vstack([error.cpu().numpy() for error in error_per_batch])
    errors = np.vstack(error_per_batch)
    errors = errors.mean(axis=0)
    
    metrics_test = evaluate_gnn(benchmark, thetas_pred=predictions, thetas_obs=observations, dataset="test")
    
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(obj=metrics_test, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

