import torch
import math
from .utils import check_bound, eval_penalty, init_population

class HCDJ:
    def __init__(self, objf, cons, LB, UB, popsize, totalgen, alpha, mpercent=0.2, kpercent=0.3, Sigma=0.2, F_val=0.7, device=torch.device('cpu')):
        self.objf = objf
        self.cons = cons
        self.LB = LB.to(device)
        self.UB = UB.to(device)
        self.popsize = popsize
        self.totalgen = totalgen
        self.alpha = alpha
        self.mpercent = mpercent
        self.kpercent = kpercent
        self.Sigma = Sigma
        self.F_val = F_val
        self.device = device
        self.nvars = LB.shape[0]

    def optimize(self):
        # Initialize population
        x = init_population(self.popsize, self.nvars, self.LB, self.UB, self.device)
        fit_x = eval_penalty(self.objf, self.cons, x, self.alpha)
        
        history = []
        
        for gen in range(self.totalgen):
            best_idx = torch.argmin(fit_x)
            worst_idx = torch.argmax(fit_x)
            best_organism = x[best_idx].unsqueeze(0)
            worst_organism = x[worst_idx].unsqueeze(0)
            
            F = self.F_val
            if F == 0.8:
                F = 0.7 + 0.3 * torch.rand(1).item()
            Sigma = self.Sigma
            if Sigma == 0.8:
                Sigma = 0.7 + 0.3 * torch.rand(1).item()
                
            rand_indices = torch.argsort(torch.rand(self.popsize, self.popsize, device=self.device), dim=1)
            row_idx = torch.arange(self.popsize, device=self.device).unsqueeze(1)
            mask = (rand_indices != row_idx)
            valid_indices = rand_indices[torch.arange(self.popsize).unsqueeze(1), mask.nonzero()[:, 1].view(self.popsize, -1)]
            
            r1 = valid_indices[:, 0]
            r2 = valid_indices[:, 1]
            r3 = valid_indices[:, 2]
            r4 = valid_indices[:, 3]
            r5 = valid_indices[:, 4]
            
            v = torch.zeros_like(x)
            
            # Divide population into 3 parts
            idx1 = int(round(self.mpercent * self.popsize))
            idx2 = int(round((self.mpercent + self.kpercent) * self.popsize))
            
            for i in range(self.popsize):
                rand_val = torch.rand(1).item()
                if i < idx1:
                    if rand_val > Sigma:
                        v[i] = best_organism[0] + F * (x[r1[i]] - x[r2[i]])
                    else:
                        v[i] = best_organism[0] + F * (x[r1[i]] - x[r2[i]]) + F * (x[r3[i]] - x[r4[i]])
                elif i < idx2:
                    if rand_val > Sigma:
                        v[i] = x[r1[i]] + F * (best_organism[0] - x[r1[i]]) + F * (x[r2[i]] - x[r3[i]])
                    else:
                        # Jaya
                        v[i] = x[r1[i]] + F * (best_organism[0] - torch.abs(x[r2[i]])) - F * (x[r3[i]] - torch.abs(worst_organism[0]))
                else:
                    if rand_val > Sigma:
                        v[i] = x[r1[i]] + F * (x[r2[i]] - x[r3[i]])
                    else:
                        v[i] = x[r1[i]] + F * (x[r2[i]] - x[r3[i]]) + F * (x[r4[i]] - x[r5[i]])
                        
            # Bound check
            v = check_bound(v, self.LB, self.UB)
            
            # Crossover (DE crossover)
            CR = 0.7 + 0.3 * torch.rand(self.popsize, 1, device=self.device)
            t = (torch.rand(self.popsize, self.nvars, device=self.device) < CR).float()
            
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
