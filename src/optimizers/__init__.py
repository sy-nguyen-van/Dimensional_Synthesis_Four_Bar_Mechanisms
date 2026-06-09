from .de import DE
from .jaya import Jaya
from .hcdj import HCDJ
from .drl import DRLOptimizer

def get_optimizer(algo_name: str, objf, cons, LB, UB, popsize, totalgen, alpha, device):
    if algo_name == 'DE':
        return DE(objf, cons, LB, UB, popsize, totalgen, alpha, device=device)
    elif algo_name == 'Jaya':
        return Jaya(objf, cons, LB, UB, popsize, totalgen, alpha, device=device)
    elif algo_name == 'HCDJ':
        return HCDJ(objf, cons, LB, UB, popsize, totalgen, alpha, device=device)
    elif algo_name == 'DRL':
        return DRLOptimizer(objf, cons, LB, UB, popsize, totalgen, alpha, device=device)
    else:
        raise ValueError(f"Unknown optimizer {algo_name}")
