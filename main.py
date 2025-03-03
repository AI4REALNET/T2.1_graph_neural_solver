import json
import pathlib
from lips.benchmark.powergridBenchmark import PowerGridBenchmark
from gnn_powergrid.gnn.gnn_simulator import GnnSimulator
from gnn_powergrid.gnn.models.gnn import GPGmodel
from gnn_powergrid import NpEncoder

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


    # use this if you have already generated some data to import them and comment two previous statements
    # benchmark = PowerGridBenchmark(benchmark_path=DATA_PATH,
    #                             benchmark_name="Benchmark4",#"DoNothing",
    #                             load_data_set=True,
    #                             config_path=BENCH_CONFIG_PATH,
    #                             log_path=LOG_PATH)


    # read the hyperparameters from the configuration file
    SIM_CONFIG_PATH = path / "configs" / "gnn.ini"
    gnn_simulator = GnnSimulator(model=GPGmodel,
                                name="gnn_torch",
                                sim_config_path=SIM_CONFIG_PATH,
                                input_size=2,
                                output_size=1,
                                epochs=EPOCHS)

    gnn_simulator.train(benchmark.train_dataset, benchmark.val_dataset)

    predictions = gnn_simulator.predict(dataset=benchmark._test_dataset)

    metrics_test = benchmark.evaluate_simulator(dataset="all", augmented_simulator=gnn_simulator)
    print(metrics_test)
    
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(obj=metrics_test, fp=f, indent=4, sort_keys=True, cls=NpEncoder)