import torch

def grashof_condition(x):
    """
    Evaluates Grashof condition for a population x of shape (popsize, nvars).
    r1, r2, r3, r4 are x[:, 0], x[:, 1], x[:, 2], x[:, 3].
    Condition: sum of max and min link lengths <= sum of other two links.
    Returns a tensor of shape (popsize,) with 0 if condition met, 1 otherwise.
    """
    links = x[:, 0:4]
    max_val, _ = torch.max(links, dim=1)
    min_val, _ = torch.min(links, dim=1)
    sum_max_min = max_val + min_val
    sum_all = torch.sum(links, dim=1)
    other = sum_all - sum_max_min
    
    # MATLAB uses: if summaxmin - Other <= 0 then 0 else 1
    violations = (sum_max_min - other > 0).float()
    return violations

def angle_sequence_violation(x, num_angles):
    """
    Evaluates angle sequence condition (angles must be cyclically increasing).
    Angles are assumed to start at index 9 (Python index) and go up to 9 + num_angles - 1.
    """
    # angles shape: (popsize, num_angles)
    angles = x[:, 9:9+num_angles]
    popsize = x.shape[0]
    
    # Find index of minimum angle for each individual
    # imin shape: (popsize,)
    _, imin = torch.min(angles, dim=1)
    
    violations = torch.zeros(popsize, device=x.device)
    
    for p in range(popsize):
        # Cyclically shift angles so that the minimum is at the start
        idx = imin[p].item()
        shifted_angles = torch.cat((angles[p, idx:], angles[p, :idx]))
        
        # Count how many times shifted_angles[i] >= shifted_angles[i+1]
        diffs = shifted_angles[1:] - shifted_angles[:-1]
        violations[p] = torch.sum(diffs <= 0).float()
        
    return violations

