from pycu_spins import CudaSpins, np, cusp
from utils import ResultsHandler
import matplotlib.pyplot as plt

import pandas as pd

def main():

    system_params = {
                "num_bodies": 256,
                "dt": 10e-4,
                "tsteps": 10000, 
                "save_step":1, #10
                "gamma": 1.0,
                "jx": 0.9, 
                "jy":0.5,
                "jz": 1.0,
                "save_start":0,
                'sub_system_size': 16,
                "model":2,
                }
 
    gpu_settings = {
            "block_size":256,
            "use_host_mem": True}

    # res_handl = ResultsHandler(system_params)
    for _ in range(100):
        cs = CudaSpins(params=system_params, gpu_params=gpu_settings)
        cs.run()
        del cs


if __name__ == '__main__':
    main()