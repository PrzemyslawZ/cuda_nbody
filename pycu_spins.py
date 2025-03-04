import cuda_interface as cusp
import numpy as np
import datetime

class Randomizer:

    """
        Interface for generating Box-Muller transformation for random input for 
        N-body spin simulation
    """

    def __init__(self, size: int, seed: int):

        """
            Initialization of random input.

            Args:
                size (int): number of random samples to be generated 
                (len of results and imput buffer)
                
                seed (int): seed for uniform distribution
        """

        np.random.seed(seed=seed)
        self.size = 2*size
        self.box_muller = cusp.floatArray(self.size)
        self._box_muller()

    def __del__(self):
        del self._u1, self._u2
        del self._z0, self.box_muller

    def _box_muller(self):
    
        """
           Box-Muller transformation gerneration
        
           Return:
                None
        """
        self._generate()
        r = np.sqrt(-2*np.log(self._u1))
        theta = 2*np.pi*self._u2
        self.b_mul_ch = r * np.cos(theta)
        self._move(self.b_mul_ch , self.box_muller)

    def _move(self, source_buffer, destination_buffer:list, fact:int = 1):
        """
            Method copying data from carray (floatAray) to Python list
            
            Args:
                source_buffer (SwigObject): source data to be copied
                destination buffer (list): destination list to which we copy data
            
            Return:
                None
        """
        for i in range(self.size*fact):
            destination_buffer[i] = source_buffer[i]
    
    def _generate(self):
        """
            Method generating uniform distribution samples
            
            Args:
                source_buffer (SwigObject): source data to be copied
                destination buffer (list): destination list to which we copy data
            
            Return:
                None
        """
        self._u1 = np.random.uniform(size=self.size)
        self._u2 = np.random.uniform(size=self.size)

class CudaSpins(Randomizer):
    
    """
        Interface for communication with CUDA code. Class inherits all method 
        from Randomizer class.
    """

    def __init__(self, params:dict={}, gpu_params:dict={}, seed:int=1337):

        """
            Initialization of spin simulation on GPU.

            Args:
                params (dict): physical parameters for spin simulations 
                gpu_params (dict): kernel parameters for size of GPU computational
                grid
                
                seed (int): seed for uniform distribution
        """

        self.params = params
        self.gpu_params = gpu_params

        self._load_default_params()
        super().__init__(self.params["num_bodies"], seed)
        self._prepare_results_buffer()

    def __del__(self):
        del self.params, self.gpu_params, self.results

    def get_platform_options(self)->None:
        """
            Method informs about possible computation platforms on which 
            user may run the simulation.
    
            Return:
                None
        """
        print("Available simulation platforms: CPU, GPU, BOTH")

    def get_results(func):
        """
            Method adjusts results buffer returned from C code
        """
        def wrapper(self, *args):
            func(self, *args)
            # TODO: Check if shift is okay AND adjust it both in python and CPP code
            # del self._results_c
            shift = self.params["tsteps"] // self.params["save_step"]
            self._move(self._results_c, self.results, shift)
            if cusp.cvar.PLATFORM == "BOTH":
                self.results = {
                                "res_gpu": self.results[:shift], 
                                "res_cpu": self.results[shift:]
                               }
            else:
                pass
        return wrapper

    @get_results
    def run(self,input_buffer:list=None)->None:
        """
            Method calls C simulation function and outputs perormance mesured 
            directly within C code.

            Args:
                input_buffer (list): input for spin simulation (initial values)

            Return:
                None
        """
        start = datetime.datetime.now()
        cusp.simulate(self.box_muller if input_buffer is None else input_buffer, 
                                self._results_c,
                                self.params, 
                                self.gpu_params)
        end = datetime.datetime.now()
        self.elapsed_time = (end-start).total_seconds() * 1000
        self._get_performance()

    def set_platform(self, platform:str ='CPU')->None:
        """
            Method sets platform on which simulation will be performed

            Args:
                paltform (str): platform type

            Return:
                None
        """
        cusp.cvar.PLATFORM = platform
        self._get_platform_info()

    def _get_performance(self)->None:
        """
            Method prints time elaspsed during simulation mesured directlty within
            C code

            Args:
                None

            Return:
                None
        """
        if cusp.cvar.PLATFORM == "CPU" or cusp.cvar.PLATFORM == "BOTH":
            print("CPU simulator elapsed time: ", cusp.cvar.TIME_CPU, " [ms]")
        if cusp.cvar.PLATFORM == "GPU" or cusp.cvar.PLATFORM == "BOTH":
            print("GPU simulator elapsed time: ", cusp.cvar.TIME_GPU, " [ms]")

        sim_time = cusp.cvar.TIME_CPU if cusp.cvar.PLATFORM == "CPU" else cusp.cvar.TIME_GPU
        print("Python elapsed time = {:.3f} [ms], ratio = {:.3f}".format(self.elapsed_time,  
        self.elapsed_time / (sim_time)))

    @staticmethod    
    def _get_platform_info()->None:
        """
            Method prints enabled computation platfrom

            Args:
                None

            Return:
                None
        """
        print(f"[INFO] Platform enabled: {cusp.cvar.PLATFORM}")

    def _prepare_results_buffer(self)->None:
        """
            Method adjusts size of results buffer to which simulation outcome 
            will be saved

            Args:
                None

            Return:
                None
        """
        # TODO: Check if size of spins is correct
        platfrom_fact = 4 if cusp.cvar.PLATFORM == "BOTH" else 1
        buffer_size = platfrom_fact * \
            self.size * \
            (self.params["tsteps"] // self.params["save_step"] - self.params["save_start"])
        self.results = np.zeros(buffer_size, dtype=np.float32)
        self._results_c = cusp.floatArray(buffer_size)
    
    def _load_default_params(self)->None:
        """
            Method creates default parameters dictionaies for both simulation 
            and the device.

            Args:
                None

            Return:
                None
        """
        if self.params == {}:
           self.params = {
                "num_bodies": 1024,
                "dt": 0.001,
                "tsteps": 1000,
                "save_step":100,
                "gamma": 1.0,
                "jx": 0.9,
                "jy": 0.9,
                "jz": 1,
                "save_start":0
                }
        if self.gpu_params == {}:
           self.gpu_params = {
            "block_size": 256, 
            "use_host_mem": False}