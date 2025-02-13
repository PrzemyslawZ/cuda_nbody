import cuda_spins
import numpy as np

class CudaSpins:
    
    def __init__(self, params: dict={}, gpu_params: dict={}):        
        self.params = params
        self.gpu_params = gpu_params

        self._load_default_params()
        self._prepare_results_buffer()

    def __del__(self):
        del cuda_spins
        del self.params, self.gpu_params, self.results

    def get_platform_options(self):
        print("Available simulation platforms: CPU, GPU, BOTH")

    def run(self,input_buff:np.array):
        if input_buff is not None:
            cuda_spins.simulate(input_buff, 
                                self.results, 
                                self.params, 
                                self.gpu_params)
            self._get_performance()
        else:
            print("[INPUT ERROR]: Provide correct input array (not None)")

    def set_platform(self, platform:str ='CPU'):
        cuda_spins.PLATFORM = platform
        self._get_platform_info()

    @staticmethod
    def _get_performance():
        if cuda_spins.PLATFORM == "CPU" or cuda_spins.PLATFORM == "BOTH":
            print("CPU simulator elapsed time: ", cuda_spins.TIME_CPU)
        if cuda_spins.PLATFORM == "GPU" or cuda_spins.PLATFORM == "BOTH":
            print("GPU simulator elapsed time: ", cuda_spins.TIME_GPU)

    @staticmethod    
    def _get_platform_info():
        print(f"[INFO] Platform enabled: {cuda_spins.PLATFORM}")

    def _prepare_results_buffer(self):
        # TODO: Check if size of spins is correct
        platfrom_fact = 2 if cuda_spins.PLATFORM == "BOTH" else 1
        self.results = np.zeros(platfrom_fact*int(self.params["num_bodies"]), 
                                dtype=np.float32)
    
    def _load_default_params(self):
        if self.params == {}:
           self.params = {
                "system_type": 16,
                "num_bodies": 1024,
                "dt": 0.001,
                "tsteps": 1000,
                "save_step":100,
                "nx_spins": 64,
                "ny_spins": 64,
                "nz_spins": 1,
                "gamma1": 0,
                "gamma2": 0,
                "nth1": 0,
                "nth2": 0,
                "gamma": 1,
                "gamma_phase": 0,
                "jx": 0.5,
                "jy": 0.2,
                "jz": 0.7,
                "Omega_x": 0,
                "Omega_y": 0,
                "Omega_z": 0,
                "Omega_D": 0
                }
        if self.gpu_params == {};
           self.gpu_params = {"block_size": 254, "blocks_num": 4}












