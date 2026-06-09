from .case1 import Case1
from .case2 import Case2
from .case3 import Case3
from .case4 import Case4
from .case5 import Case5

def get_case_functions(case_name: str):
    cases = {
        'Case1': Case1,
        'Case2': Case2,
        'Case3': Case3,
        'Case4': Case4,
        'Case5': Case5
    }
    
    if case_name not in cases:
        raise ValueError(f"Unknown case {case_name}")
        
    return cases[case_name].objf, cases[case_name].cons
