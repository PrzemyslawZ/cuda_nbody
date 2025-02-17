import cuda_interface as cusp
import numpy as np


class Randomizer:

    def __init__(self, size, seed):
        np.random.seed(seed=seed)
        self.size = size
        self.box_muller = cusp.floatArray(self.size)
        self._box_muller()

    def __del__(self):
        del self._u1, self._u2
        del self._z0, self.box_muller

    def _box_muller(self):
        self._generate()
        r = np.sqrt(-2*np.log(self._u1))
        theta = 2*np.pi*self._u2
        self._move(r* np.cos(theta), self.box_muller)

    def _move(self, source_buffer, destination_buffer):
        for i in range(self.size):
            destination_buffer[i] = source_buffer[i]
    
    def _generate(self):
        self._u1 = np.random.uniform(size=self.size)
        self._u2 = np.random.uniform(size=self.size)

class CudaSpins(Randomizer):
    
    def __init__(self, params: dict={}, gpu_params: dict={}, seed:int =521):     
        self.params = params
        self.gpu_params = gpu_params

        self._load_default_params()
        super().__init__(self.params["num_bodies"], seed)
        self._prepare_results_buffer()

    def __del__(self):
        del self.params, self.gpu_params, self.results

    def get_platform_options(self):
        print("Available simulation platforms: CPU, GPU, BOTH")

    def get_results(func):
        def wrapper(self, *args):
            func(self, *args)
            # TODO: Check if shift is okay AND adjust it both in python and CPP code
            self._move(self._results_c, self.results)
            shift = self.params["tsteps"] // self.params["save_step"]
            if cusp.cvar.PLATFORM == "BOTH":
                self.results = {
                                "res_gpu": self.results[:shift], 
                                "res_cpu": self.results[shift:]
                               }
            else:
                pass
        return wrapper

    @get_results
    def run(self,input_buffer:np.array=None):
        cusp.simulate(self.box_muller if input_buffer is None else input_buffer, 
                                self._results_c,
                                self.params, 
                                self.gpu_params)
        self._get_performance()

    def set_platform(self, platform:str ='CPU'):
        cusp.cvar.PLATFORM = platform
        self._get_platform_info()

    @staticmethod
    def _get_performance():
        if cusp.cvar.PLATFORM == "CPU" or cusp.cvar.PLATFORM == "BOTH":
            print("CPU simulator elapsed time: ", cusp.cvar.TIME_CPU)
        if cusp.cvar.PLATFORM == "GPU" or cusp.cvar.PLATFORM == "BOTH":
            print("GPU simulator elapsed time: ", cusp.cvar.TIME_GPU)

    @staticmethod    
    def _get_platform_info():
        print(f"[INFO] Platform enabled: {cusp.cvar.PLATFORM}")

    def _prepare_results_buffer(self):
        # TODO: Check if size of spins is correct
        platfrom_fact = 2 if cusp.cvar.PLATFORM == "BOTH" else 1
        self.results = np.zeros(platfrom_fact*self.size, dtype=np.float32)
        self._results_c = cusp.floatArray(platfrom_fact * self.size)
    
    def _load_default_params(self):
        if self.params == {}:
           self.params = {
                "system_type": 16,
                "num_bodies": 32,
                "dt": 0.001,
                "tsteps": 1000,
                "save_step":10,
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
        if self.gpu_params == {}:
           self.gpu_params = {"block_size": 254, "blocks_num": 4}