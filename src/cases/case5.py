import torch
from src.cases.common import grashof_condition, angle_sequence_violation

class Case5:
    Cdx = [20.0, 17.66, 11.736, 5.0, 0.60307, 0.60307, 5.0, 11.736, 17.66, 20.0]
    Cdy = [10.0, 15.142, 17.878, 16.928, 12.736, 7.2638, 3.0718, 2.1215, 4.8577, 10.0]
    
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
        
        x7 = x[:, 6]
        x8 = x[:, 7]
        x9 = x[:, 8]
        
        Cdx_t = torch.tensor(Case5.Cdx, device=x.device)
        Cdy_t = torch.tensor(Case5.Cdy, device=x.device)
        
        for i in range(len(Case5.Cdx)):
            teta2 = x[:, 9 + i]
            K1 = r1 / (r2 + 1e-12)
            K4 = r1 / (r3 + 1e-12)
            K5 = (r4**2 - r1**2 - r2**2 - r3**2) / (2 * r3 * r2 + 1e-12)
            
            D = torch.cos(teta2) - K1 + K4 * torch.cos(teta2) + K5
            E = -2 * torch.sin(teta2)
            F = K1 + (K4 - 1) * torch.cos(teta2) + K5
            
            discriminant = E**2 - 4 * D * F
            discriminant_c = torch.complex(discriminant, torch.zeros_like(discriminant))
            D_c = torch.complex(D, torch.zeros_like(D))
            E_c = torch.complex(E, torch.zeros_like(E))
            
            ATAN3 = (-E_c - torch.sqrt(discriminant_c)) / (2 * D_c + 1e-12)
            teta3 = 2 * torch.atan(ATAN3).real
            
            Crx = r2 * torch.cos(teta2) + rcx * torch.cos(teta3) - rcy * torch.sin(teta3)
            Cry = r2 * torch.sin(teta2) + rcx * torch.sin(teta3) + rcy * torch.cos(teta3)
            
            C_x = torch.cos(x9) * Crx - torch.sin(x9) * Cry + x7
            C_y = torch.sin(x9) * Crx + torch.cos(x9) * Cry + x8
            
            f += (C_x - Cdx_t[i])**2 + (C_y - Cdy_t[i])**2
            
        return f

    @staticmethod
    def cons(x):
        links = x[:, 0:4]
        min_val, _ = torch.min(links, dim=1)
        summaxmin = torch.max(links, dim=1)[0] + min_val
        other = torch.sum(links, dim=1) - summaxmin
        vo1 = (summaxmin - other > 0).float()
        
        vo2 = angle_sequence_violation(x, num_angles=10)
        
        return vo1 + vo2
