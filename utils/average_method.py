import numpy as np
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.getcwd())
from utils import methods, test_functions

"""
this should be multithreaded, honeslty so should smc in places but there
is nothing stopping every method_obj instance running on its own thread
they all share nothing, we would just need a pause ocndition before summing
"""
class AverageMethod():
    def __init__(self, 
                 method_class : methods.DefaultMethodClass,
                 func_class : test_functions.FuncToMinimise,
                 n_method_instances : int = 30):
        
        # also want to take an object that i can duplicate here instead of a class
        self.method_objs = [method_class(func_class = func_class) for _ in range(n_method_instances)]

        # the part that should be mutlithreaded:
        for meth in self.method_objs:
            meth.run()

        self.n_instances = n_method_instances
        self.sum_regret = []
        self.get_sum_regret()

        self.mean_regret = self.sum_regret / n_method_instances

    def get_sum_regret(self):
        self.sum_regret = self.method_objs[0].get_regret()
        for meth in self.method_objs[1:]:
            self.sum_regret += meth.get_regret()

    def get_result(self):
        return {'name': self.method_objs[0].method_name,
                'regret':self.mean_regret,
                'n_instances':self.n_instances}
    
def main():
    AverageMethod(methods.NormalGp, 
                  func_class=test_functions.Branin, 
                  n_method_instances=30)
    return

if __name__ == '__main__':
    main()