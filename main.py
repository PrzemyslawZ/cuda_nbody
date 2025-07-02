from pycu_spins import CudaSpins, np, cusp
from utils import ResultsHandler
import matplotlib.pyplot as plt

import pandas as pd

def main():

    model = 'xyz'
    system_params = {
                "num_bodies": 256,
                "dt": 1e-3, 
                "tsteps": 1000, #20000  
                "save_step":1, #10
                "gamma": 1.0,
                "jx": 0.9,
                "jy": 0.9,
                "jz": 1.0,
                "save_start":0,
                'sub_system_size': 256,
                "model":model,
                }
 
    gpu_settings = {
            "block_size":256,
            "use_host_mem": True}

    
    size = system_params['num_bodies']
    sx = np.zeros(size)
    sy = np.zeros(size)
    sz = np.ones(size)

    initial_state = [0,0,1]*size # np.concatenate([sx, sy, sz])

    cs = CudaSpins(
        params=system_params, 
        gpu_params=gpu_settings, 
        initial_state=initial_state)

    res_handl = ResultsHandler(system_params)
    # for _ in range(100):
    #     cs = CudaSpins(params=system_params, gpu_params=gpu_settings)
    cs.run()
    # print(initial_state)
    print(cs.results.size, cs.results, np.max(cs.results))
    res_handl.get_momentum(cs.results)
    res_handl.plot(f'./{model}')
    
    #     del cs

if __name__ == '__main__':
    main()