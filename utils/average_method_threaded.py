import numpy as np
import matplotlib.pyplot as plt
import copy
import threading

import os, sys
sys.path.append(os.getcwd())
from utils import methods, test_functions

"""
might actually be slower on my laptop for 6 instances hopefully its faster
for 30 instances at 200 func evals per instance
"""
class AverageMethod():
    def __init__(self, 
                 method_obj : methods.DefaultMethodClass,
                 n_method_instances : int = 30):
        
        # also want to take an object that i can duplicate here instead of a class
        self.method_objs = [copy.deepcopy(method_obj) for _ in range(n_method_instances)]

        # the part that should be mutlithreaded:
        threads = []
        for meth in self.method_objs:
            thread = threading.Thread(target=meth.run)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()

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
    
    def print_individual_regret(self):
        for meth in self.method_objs:
            print(meth.get_regret())
    
def main():
    am = AverageMethod(method_obj = 
            methods.NormalGp(func_class=test_functions.Branin,
                            n_iters=30), 
                        n_method_instances=30)
    am.print_individual_regret()
    return

if __name__ == '__main__':
    main()