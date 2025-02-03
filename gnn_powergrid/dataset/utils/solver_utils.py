import grid2op
from grid2op.Parameters import Parameters
from lightsim2grid.lightSimBackend import LightSimBackend


def get_obs(benchmark):
    """Get an observation of a Grid2op environment contained in a benchmark object

    Args:
        benchmark (_type_): LIPS benchmark

    Returns:
        _type_: Grid2op environment and an observation
    """
    params = Parameters()
    params.ENV_DC = True
    env = grid2op.make(benchmark.env_name, param=params, backend=LightSimBackend())
    obs = env.reset()
    return env, obs

def get_obs_with_config(config):
    """Get an observation of a Grid2op environment contained in a benchmark object

    Args:
        benchmark (_type_): LIPS benchmark

    Returns:
        _type_: Grid2op environment and an observation
    """
    params = Parameters()
    params.ENV_DC = True
    env = grid2op.make(config["env_name"], param=params, backend=LightSimBackend())
    obs = env.reset()
    return env, obs
