import torch
import math

def get_config(case_name: str, device: torch.device = torch.device('cpu')):
    """Returns nvars, LB, UB, Popsize, Totalgen, alpha for a given case."""
    if case_name == 'Case1':
        nvars = 15
        Popsize = 100
        Totalgen = 1000
        alpha = 1e6
        L1 = torch.zeros(4, device=device)
        H1 = 60 * torch.ones(4, device=device)
        L2 = -60 * torch.ones(4, device=device)
        H2 = 60 * torch.ones(4, device=device)
        L3 = torch.zeros(7, device=device)
        H3 = 2 * math.pi * torch.ones(7, device=device)
        LB = torch.cat([L1, L2, L3])
        UB = torch.cat([H1, H2, H3])
        
    elif case_name == 'Case2':
        nvars = 6
        Popsize = 50
        Totalgen = 100
        alpha = 1e3
        L1 = torch.zeros(4, device=device)
        H1 = 50 * torch.ones(4, device=device)
        L2 = -50 * torch.ones(2, device=device)
        H2 = 50 * torch.ones(2, device=device)
        LB = torch.cat([L1, L2])
        UB = torch.cat([H1, H2])

    elif case_name == 'Case3':
        nvars = 10
        Popsize = 50
        Totalgen = 100
        alpha = 1e3
        L1 = torch.zeros(4, device=device)
        H1 = 50 * torch.ones(4, device=device)
        L2 = -50 * torch.ones(4, device=device)
        H2 = 50 * torch.ones(4, device=device)
        L3 = torch.zeros(2, device=device)
        H3 = 2 * math.pi * torch.ones(2, device=device)
        LB = torch.cat([L1, L2, L3])
        UB = torch.cat([H1, H2, H3])

    elif case_name == 'Case4':
        nvars = 9
        Popsize = 50
        Totalgen = 1000
        alpha = 1e4
        L1 = torch.zeros(4, device=device)
        H1 = 50 * torch.ones(4, device=device)
        L2 = -50 * torch.ones(4, device=device)
        H2 = 50 * torch.ones(4, device=device)
        L3 = torch.tensor([0.0], device=device)
        H3 = torch.tensor([2 * math.pi], device=device)
        LB = torch.cat([L1, L2, L3])
        UB = torch.cat([H1, H2, H3])

    elif case_name == 'Case5':
        nvars = 19
        Popsize = 100
        Totalgen = 1000
        alpha = 1e6
        L1 = torch.zeros(4, device=device)
        H1 = 80 * torch.ones(4, device=device)
        L2 = -80 * torch.ones(4, device=device)
        H2 = 80 * torch.ones(4, device=device)
        L3 = torch.zeros(11, device=device)
        H3 = 2 * math.pi * torch.ones(11, device=device)
        LB = torch.cat([L1, L2, L3])
        UB = torch.cat([H1, H2, H3])
    else:
        raise ValueError(f"Unknown case {case_name}")

    return nvars, LB, UB, Popsize, Totalgen, alpha
