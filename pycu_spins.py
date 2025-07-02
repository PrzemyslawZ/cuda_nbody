import cuda_interface as cusp
import pandas as pd
import numpy as np
import datetime

class Initializer:

    """
        Interface generates or loading initial state for N-body spin simulation
    """

    def __init__(self, model:int, size: int, seed: int, file: str=None, initial_state: np.array=None)->None:
        """
            Initialization of initial input.

            Args:
                model (int) - type of  model we want to simulate
                size (int): number of random samples to be generated (len of results and imput buffer)
                seed (int): seed for uniform distribution
                file (str): initial state txt file
                initial_state (np.array): initial state for simulation
        """
        self.size = 3 * size if model == 2 else 2 * size
        self.initial_state = cusp.floatArray(self.size)

        # print(initial_state)

        if file is None:
            self._generate_initial_state(seed)
        else:
            self._load_initalize_state(file)

        if initial_state is not None:
            self.generated_state = initial_state

        self._move(self.generated_state , self.initial_state)

    def __del__(self):
        del self.initial_state, self.generated_state

    def _generate_initial_state(self, seed)->None:
        """
           Method generating uniform distribution samples - initial state
        
           Return:
                None
        """

        np.random.seed(seed=seed)
        # self.generated_state = np.concatenate([
        #     np.random.uniform(size=self.size // 2, low=0.0, high=2*np.pi),
        #     np.random.uniform(size=self.size // 2, low=0.0, high=np.pi)
        #     ])
        theta = np.random.uniform(size=self.size // 2, low=0.0, high=np.pi)
        phi = np.random.uniform(size=self.size // 2, low=0.0, high=2*np.pi)
        self.generated_state = np.concatenate(list(zip(theta, phi)))

    def _move(self, source_buffer, destination_buffer:list, fact:int = 1)->None:
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

    def _load_data(self, file: str)->None:
        """
            Method loads files with initial state into data frame

            Args:
                file (str): initial state txt file

            Return:
                None
        """

        try:
            self.df = pd.read_csv(file, names=["sx", "sy", "sz"]).astype(np.float32)
        except:
            exit("[ERROR] Cannot open the file, (3 separated columns expected)")

    def _load_initalize_state(self, file: str)->None: 
        """
            Method computes phi and theta parameter

            Args:
                file (str): initial state txt file

            Return:
                None
        """
        self._load_data(file)
        self.df['theta'] = self.df['sz'].apply(lambda x: -x / np.sqrt(3))
        self.df['phi'] = (np.arctan(-self.df['sx'] / self.df['sy']) + (self.df['sy'] > 0) * np.pi) % (2 * np.pi)
        self.generated_state = np.concatenate(list(zip(self.df['theta'], self.df['phi']))) #np.concatenate([self.df['theta'], self.df['phi']])

class CudaSpins(Initializer):
    
    """
        Interface for communication with CUDA code. Class inherits all method 
        from Initializer class.
    """

    def __init__(self, params:dict={}, gpu_params:dict={}, seed:int=1337, file:str=None, initial_state: np.array=None):
        """
            Initialization of spin simulation on GPU.

            Args:
                params (dict): physical parameters for spin simulations 
                gpu_params (dict): kernel parameters for size of GPU computational grid
                seed (int): seed for uniform distribution
                file (str): initial state txt file
                initial_state (np.array): initial state for simulation
        """

        self.params = params
        self.gpu_params = gpu_params

        self._load_default_params()
        super().__init__(
            self.params["model"], self.params["num_bodies"], 
            seed, file, initial_state)
        self._prepare_results_buffer()

    def __del__(self):
        del self.params, self.gpu_params, self.results

    def get_results(func):
        """
            Method adjusts results buffer returned from C code
        """
        def wrapper(self, *args):
            func(self, *args)
            shift = (self.params["tsteps"] - self.params["save_start"]) // self.params["save_step"]
            self._move(self._results_cuda, self.results, shift)
        return wrapper

    @get_results
    def run(self)->None:
        """
            Method calls C simulation function and outputs perormance mesured 
            directly within C code.

            Args:
                None

            Return:
                None
        """

        start = datetime.datetime.now()
        cusp.simulate(self.initial_state, 
                        self._results_cuda,
                        self.params, 
                        self.gpu_params)
        
        end = datetime.datetime.now()
        self.elapsed_time = (end-start).total_seconds() * 1000
        self._get_performance()

    def _get_performance(self)->None:
        """
            Method prints time elaspsed during simulation mesured directlty within
            C code

            Args:
                None

            Return:
                None
        """

        print("Python elapsed time = {:.3f} [ms] GPU time = {:.3f} [ms], ratio = {:.3f}".format(
            self.elapsed_time, cusp.cvar.TIME_GPU,
            self.elapsed_time / (cusp.cvar.TIME_GPU)))
        self.platf_time = cusp.cvar.TIME_GPU

    def _prepare_results_buffer(self)->None:
        """
            Method adjusts size of results buffer to which simulation outcome 
            will be saved

            Args:
                None

            Return:
                None
        """

        buffer_size = self.size * \
            ((self.params["tsteps"] - self.params["save_start"]) // self.params["save_step"])
        self.results = np.zeros(buffer_size, dtype=np.float32)
        self._results_cuda = cusp.floatArray(buffer_size)

    def _choose_model(self) -> None:
        if self.params["model"] == 'xy':
            self.params["model"] = 1
        elif self.params["model"] == 'xyz':
            self.params["model"] = 2
        else:
            self.params["model"] = 3

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
                "save_start":0,
                'sub_system_size': 32,
                "model": 1,
                }
        else:
            self._choose_model()

        if self.gpu_params == {}:
           self.gpu_params = {
            "block_size": 256, 
            "use_host_mem": False}