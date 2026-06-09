import torch
import math
from src.cases.common import grashof_condition

class Case2:
    Cdx = [3, 2.759, 2.372, 1.890, 1.355]
    Cdy = [3, 3.363, 3.663, 3.862, 3.943]
    Teta2 = [math.pi/6, math.pi/4, math.pi/3, 5*math.pi/12, math.pi/2]
    
    @staticmethod
    def objf(x):
        popsize = x.shape[0]
        f = torch.zeros(popsize, device=x.device)
        
        r1 = x[:, 0]
        r2 = x[:, 1]
        r3 = x[:, 2]
        r4 = x[:, 3]
        rcx = x[:, 4]
        rcy = x[:, 5]
        
        Cdx_t = torch.tensor(Case2.Cdx, device=x.device)
        Cdy_t = torch.tensor(Case2.Cdy, device=x.device)
        Teta2_t = torch.tensor(Case2.Teta2, device=x.device)
        
        for i in range(len(Case2.Cdx)):
            teta2 = Teta2_t[i]
            K1 = r1 / (r2 + 1e-12)
            K4 = r1 / (r3 + 1e-12)
            K5 = (r4**2 - r1**2 - r2**2 - r3**2) / (2 * r3 * r2 + 1e-12)
            
            D = torch.cos(teta2) - K1 + K4 * torch.cos(teta2) + K5
            E = -2 * torch.sin(teta2) * torch.ones_like(D)
            F = K1 + (K4 - 1) * torch.cos(teta2) + K5
            
            discriminant = E**2 - 4 * D * F
            discriminant_c = torch.complex(discriminant, torch.zeros_like(discriminant))
            D_c = torch.complex(D, torch.zeros_like(D))
            E_c = torch.complex(E, torch.zeros_like(E))
            
            ATAN3 = (-E_c - torch.sqrt(discriminant_c)) / (2 * D_c + 1e-12)
            teta3 = 2 * torch.atan(ATAN3).real
            
            Crx = r2 * torch.cos(teta2) + rcx * torch.cos(teta3) - rcy * torch.sin(teta3)
            Cry = r2 * torch.sin(teta2) + rcx * torch.sin(teta3) + rcy * torch.cos(teta3)
            
            f += (Crx - Cdx_t[i])**2 + (Cry - Cdy_t[i])**2
            
        return f

    @staticmethod
    def cons(x):
        links = x[:, 0:4]
        min_val, _ = torch.min(links, dim=1)
        summaxmin = torch.max(links, dim=1)[0] + min_val
        other = torch.sum(links, dim=1) - summaxmin
        vo1 = (summaxmin - other > 0).float()
        return vo1
