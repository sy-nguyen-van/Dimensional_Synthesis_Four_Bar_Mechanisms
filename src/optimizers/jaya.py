import torch
from .utils import check_bound, eval_penalty, init_population

class Jaya:
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
        
        history = []
        
        for gen in range(self.totalgen):
            # Best and Worst organisms
            best_idx = torch.argmin(fit_x)
            worst_idx = torch.argmax(fit_x)
            best_organism = x[best_idx].unsqueeze(0)
            worst_organism = x[worst_idx].unsqueeze(0)
            
            # Selection for mutation
            rand_indices = torch.argsort(torch.rand(self.popsize, self.popsize, device=self.device), dim=1)
            row_idx = torch.arange(self.popsize, device=self.device).unsqueeze(1)
            mask = (rand_indices != row_idx)
            valid_indices = rand_indices[torch.arange(self.popsize).unsqueeze(1), mask.nonzero()[:, 1].view(self.popsize, -1)]
            
            r1 = valid_indices[:, 0]
            r2 = valid_indices[:, 1]
            r3 = valid_indices[:, 2]
            
            # Mutation (Jaya equation)
            rand1 = torch.rand(self.popsize, 1, device=self.device)
            rand2 = torch.rand(self.popsize, 1, device=self.device)
            
            v = x[r1] + rand1 * (best_organism - torch.abs(x[r2])) - rand2 * (worst_organism - torch.abs(x[r3]))
            
            # Bound check
            v = check_bound(v, self.LB, self.UB)
            
            # Evaluate offspring
            fit_v = eval_penalty(self.objf, self.cons, v, self.alpha)
            
            # Selection
            replace_mask = fit_v < fit_x
            
            x = torch.where(replace_mask.unsqueeze(1), v, x)
            fit_x = torch.where(replace_mask, fit_v, fit_x)
            
            best_idx = torch.argmin(fit_x)
            best_fit = fit_x[best_idx].item()
            history.append(best_fit)
            
            best_organism = x[best_idx].unsqueeze(0)
            best_cost = self.objf(best_organism).item()
            best_cons = self.cons(best_organism).item()
            print(f"Gen {gen+1}/{self.totalgen} | Cost: {best_cost:.4f} | Constraint: {best_cons:.4f} | Fit: {best_fit:.4f}")
            
        best_idx = torch.argmin(fit_x)
        return x[best_idx], fit_x[best_idx].item(), history
