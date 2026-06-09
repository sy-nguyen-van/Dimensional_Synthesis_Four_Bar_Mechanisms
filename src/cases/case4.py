import torch
import math

class Case4:
    Cdx = [0.0, 1.9098, 6.9098, 13.09, 18.09, 20.0]
    Cdy = [0.0, 5.8779, 9.5106, 9.5106, 5.8779, 0.0]
    Teta2 = [math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3, 5*math.pi/6, math.pi]
    
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
        
        Cdx_t = torch.tensor(Case4.Cdx, device=x.device)
        Cdy_t = torch.tensor(Case4.Cdy, device=x.device)
        Teta2_t = torch.tensor(Case4.Teta2, device=x.device)
        
        for i in range(len(Case4.Cdx)):
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
            
            C_x = torch.cos(x9) * Crx - torch.sin(x9) * Cry + x7
            C_y = torch.sin(x9) * Crx + torch.cos(x9) * Cry + x8
            
            f += (C_x - Cdx_t[i])**2 + (C_y - Cdy_t[i])**2
            
        return f

    @staticmethod
    def cons(x):
        links = x[:, 0:4]
        min_val, min_idx = torch.min(links, dim=1)
        r2_is_min = (min_idx == 1)
        
        summaxmin = torch.max(links, dim=1)[0] + min_val
        other = torch.sum(links, dim=1) - summaxmin
        vo1_cond = (summaxmin - other <= 0) & r2_is_min
        vo1 = (~vo1_cond).float()
        
        return vo1
