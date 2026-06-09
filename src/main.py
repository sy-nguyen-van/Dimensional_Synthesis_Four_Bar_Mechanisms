import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import time
from src.config import get_config
from src.cases import get_case_functions
from src.optimizers import get_optimizer

def main():
    parser = argparse.ArgumentParser(description="Synthesis of Four-Bar Mechanisms Optimization")
    parser.add_argument('--case', type=str, default='Case4', choices=[f'Case{i}' for i in range(1, 6)],
                        help='Case number to run')
    parser.add_argument('--algo', type=str, default='HCDJ', choices=['DE', 'Jaya', 'HCDJ'],
                        help='Optimization algorithm')
    parser.add_argument('--runs', type=int, default=1, help='Number of independent runs')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load problem setup
    nvars, LB, UB, popsize, totalgen, alpha = get_config(args.case, device)
    objf, cons = get_case_functions(args.case)

    best_overall_fval = float('inf')
    best_overall_xval = None

    for run in range(args.runs):
        print(f"--- Run {run + 1}/{args.runs} ---")
        
        optimizer = get_optimizer(
            algo_name=args.algo,
            objf=objf,
            cons=cons,
            LB=LB,
            UB=UB,
            popsize=popsize,
            totalgen=totalgen,
            alpha=alpha,
            device=device
        )
        
        start_time = time.time()
        best_x, best_fval, history = optimizer.optimize()
        elapsed_time = time.time() - start_time
        
        print(f"Time: {elapsed_time:.2f} s")
        print(f"Best fval: {best_fval}")
        
        if best_fval < best_overall_fval:
            best_overall_fval = best_fval
            best_overall_xval = best_x

    print("\n===============================")
    print(f"Optimization Finished.")
    print(f"Best Overall Cost: {best_overall_fval}")
    print(f"Best Variables: {best_overall_xval.cpu().numpy()}")
    print("===============================")

if __name__ == "__main__":
    main()
