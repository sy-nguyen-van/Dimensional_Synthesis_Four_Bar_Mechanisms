import torch

def check_bound(v, LB, UB):
    """
    Bounds checking for the population v.
    If a variable falls outside [LB, UB], it's reflected. If it still falls outside, it's clipped.
    Matches MATLAB check_bound implementation.
    """
    # Lower bound reflection
    mask_low = v < LB
    v = torch.where(mask_low, 2 * LB - v, v)
    
    # After lower bound reflection, check if it exceeds upper bound
    mask_low_upper = (v > UB) & mask_low
    v = torch.where(mask_low_upper, UB, v)
    
    # Upper bound reflection
    mask_up = v > UB
    v = torch.where(mask_up, 2 * UB - v, v)
    
    # After upper bound reflection, check if it falls below lower bound
    mask_up_low = (v < LB) & mask_up
    v = torch.where(mask_up_low, LB, v)
    
    return v

def eval_penalty(objf, cons, x, alpha):
    """
    Evaluates objective function and adds penalty for constraint violations.
    """
    f = objf(x)
    V = cons(x)
    return f + alpha * V

def init_population(popsize, nvars, LB, UB, device=torch.device('cpu')):
    """
    Initializes a population randomly within [LB, UB].
    """
    rand_vals = torch.rand(popsize, nvars, device=device)
    return LB + rand_vals * (UB - LB)
