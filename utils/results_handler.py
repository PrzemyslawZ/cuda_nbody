import numpy as np 
import matplotlib.pyplot as plt
import statistics

class ResultsHandler:

    def __init__(self, params):
        self.params = params
        
    def get_mean_values(func):
        """
            Method calculates mean value of the angular momentum
        """

        def wrapper(self, *args):
            func(self, *args)
            self.sx_mean = np.reshape(self.sx, (-1, self.params["num_bodies"])).mean(axis=-1)
            self.sy_mean = np.reshape(self.sy, (-1, self.params["num_bodies"])).mean(axis=-1)
            self.sz_mean = np.reshape(self.sz, (-1, self.params["num_bodies"])).mean(axis=-1)

        return wrapper

    @get_mean_values
    def get_momentum(self, sim_results: np.array)->None:
        """
            Method calculates angular momentum

            Args:
                sim_results (np.array) - simulation results

            Return:
                None
        """

        if self.params['model'] == 2:
            self.sx = sim_results[::3]
            self.sy = sim_results[1::3]
            self.sz = sim_results[2::3]
            # print(statistics.variance(self.sx),statistics.variance(self.sy),statistics.variance(self.sz))
        else:
            self.sx = np.sqrt(3)*np.sin(sim_results[::2]) * np.sin(sim_results[1::2])
            self.sy = -np.sqrt(3)*np.sin(sim_results[::2]) * np.cos(sim_results[1::2])
            self.sz = -np.sqrt(3)*np.cos(sim_results[::2])

    def plot(self, file_name:str)->None:
        """
            Plotting results (angular momentum)

            Args:
                file_name (str) - name of plot file

            Return:
                None
        """

        self._prepare_tlattice()
        plt.figure(figsize=(6,2.5))
        plt.subplot(121)
        plt.title("Raw data")
        plt.plot(self.time_body, self.sx, label="sx")
        plt.plot(self.time_body, self.sy, label="sy")
        plt.plot(self.time_body, self.sz, label="sz")
        plt.xlabel("t", fontsize=14)
        plt.ylabel("s", fontsize=14)
        plt.subplot(122)
        plt.title("Averaged")
        plt.plot(self.time_line, self.sx_mean, label="<sx>")
        plt.plot(self.time_line, self.sy_mean, label="<sy>")
        plt.plot(self.time_line, self.sz_mean, label="<sz>")    
        plt.xlabel("t", fontsize=14)
        plt.ylabel("<s>", fontsize=14)
        plt.legend(frameon=False, prop={"size": 8})
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"./{file_name}.png", dpi=200)
        plt.close()

    def _prepare_tlattice(self)->None:
        """
            Method prepares time lattices for plots

            Args:
                None

            Return:
                None
        """

        tsteps = (self.params["tsteps"]-self.params["save_start"]) // self.params["save_step"]
        self.time_line = np.linspace(0, self.params['tsteps'] * self.params["dt"], tsteps)
        self.time_body = np.linspace(0, self.params['tsteps'] * self.params["dt"], 
        tsteps*self.params["num_bodies"])
