import math
from typing import Union
from lips.metrics.ml_metrics.external_metrics import mape_quantile
from lips.evaluation.powergrid_evaluation import PowerGridEvaluation

from ..dataset.utils.solver_utils import get_obs
from ..dataset.utils.graph_utils import get_all_active_powers

def evaluate_gnn(benchmark, thetas_pred, thetas_obs, dataset: Union["test","ood"] ="test"):
    if dataset == "test":
        data = getattr(benchmark, "_test_dataset").data
    elif dataset == "ood":
        data = getattr(benchmark, "_test_ood_topo_dataset").data
    else:
        raise KeyError("You should select between `test` and `ood`")
    
    predictions = thetas_pred * (180/math.pi)
    metrics = {}
    metrics["theta"] = {}
    mape10 = mape_quantile(y_true=thetas_obs.detach().cpu(), y_pred=predictions.detach().cpu(), quantile=0.9)
    # print("MAPE10 on theta: ", MAPE_10)
    metrics["theta"]["MAPE10"] = mape10
    
    env, obs = get_obs(benchmark)
    my_predictions = {}
    my_predictions["p_or"], my_predictions["p_ex"] = get_all_active_powers(data,
                                                                           obs,
                                                                           theta_bus=predictions.cpu().view(-1,obs.n_sub*2))
    evaluation = PowerGridEvaluation.from_benchmark(benchmark)
    metrics["power"] = evaluation.evaluate(observations=data, 
                                           predictions=my_predictions, 
                                           env=env)
    
    return metrics
    
    
    

    
    