import torch
from .utils import check_bound, eval_penalty, init_population

class DE:
    def __init__(self, objf, cons, LB, UB, popsize, totalgen, alpha, device=torch.device('cpu')):
        self.objf = objf
        self.cons = cons
        self.LB = LB.to(device)
        self.UB = UB.to(device)
        self.popsize = popsize
        self.totalgen = totalgen
        self.alpha = alpha
        self.device = device
        self.nvars = LB.shape[0]

    def optimize(self):
        # Initialize population
        x = init_population(self.popsize, self.nvars, self.LB, self.UB, self.device)
        fit_x = eval_penalty(self.objf, self.cons, x, self.alpha)
        
        # Track history
        history = []
        
        for gen in range(self.totalgen):
            # Selection for mutation: 3 distinct random indices for each individual, not equal to i
            rand_indices = torch.argsort(torch.rand(self.popsize, self.popsize, device=self.device), dim=1)
            # Ensure we don't pick the individual itself
            row_idx = torch.arange(self.popsize, device=self.device).unsqueeze(1)
            mask = (rand_indices != row_idx)
            # Take first 3 valid indices for each row
            # Since mask has only one False per row, sorting mask (False is 0, True is 1) descending puts True first
            valid_indices = rand_indices[torch.arange(self.popsize).unsqueeze(1), mask.nonzero()[:, 1].view(self.popsize, -1)]
            
            r1 = valid_indices[:, 0]
            r2 = valid_indices[:, 1]
            r3 = valid_indices[:, 2]
            
            # F = rand for each individual
            F = torch.rand(self.popsize, 1, device=self.device)
            
            # Mutation
            v = x[r1] + F * (x[r2] - x[r3])
            
            # Bound check
            v = check_bound(v, self.LB, self.UB)
            
            # Crossover
            CR = 0.7 + 0.3 * torch.rand(self.popsize, 1, device=self.device)
            t = (torch.rand(self.popsize, self.nvars, device=self.device) < CR).float()
            
            # Ensure at least one element is taken from v
            j_rand = torch.randint(0, self.nvars, (self.popsize, 1), device=self.device)
            t.scatter_(1, j_rand, 1.0)
            
            u = t * v + (1 - t) * x
            
            # Evaluate offspring
            fit_u = eval_penalty(self.objf, self.cons, u, self.alpha)
            
            # Selection
            replace_mask = fit_u < fit_x
            
            x = torch.where(replace_mask.unsqueeze(1), u, x)
            fit_x = torch.where(replace_mask, fit_u, fit_x)
            
            best_fit = torch.min(fit_x).item()
            history.append(best_fit)
            
        best_idx = torch.argmin(fit_x)
        return x[best_idx], fit_x[best_idx].item(), history
