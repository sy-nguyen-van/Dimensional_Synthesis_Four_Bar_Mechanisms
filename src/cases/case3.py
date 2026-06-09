import torch
import math

class Case3:
    CD = [[0.5, 1.1], [0.4, 1.1], [0.3, 1.1], [0.2, 1.0], 
          [0.1, 0.9], [0.05, 0.75], [0.02, 0.6], 
          [0.0, 0.5], [0.0, 0.4], [0.03, 0.3], 
          [0.1, 0.25], [0.15, 0.2], [0.2, 0.3], 
          [0.3, 0.4], [0.4, 0.5], [0.5, 0.7], 
          [0.6, 0.9], [0.6, 1.0]]
    
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
        
        Cdx = torch.tensor([row[0] for row in Case3.CD], device=x.device)
        Cdy = torch.tensor([row[1] for row in Case3.CD], device=x.device)
        
        for i in range(len(Case3.CD)):
            teta2 = x[:, 9] + i * (math.pi / 9)
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
            
            f += (C_x - Cdx[i])**2 + (C_y - Cdy[i])**2
            
        return f

    @staticmethod
    def cons(x):
        links = x[:, 0:4]
        min_val, _ = torch.min(links, dim=1)
        summaxmin = torch.max(links, dim=1)[0] + min_val
        other = torch.sum(links, dim=1) - summaxmin
        vo1 = (summaxmin - other > 0).float()
        return vo1
