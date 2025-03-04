from pycu_spins import CudaSpins, np, cusp
import matplotlib.pyplot as plt

def main():
    
    cs = CudaSpins(
        params = 
                {
                "num_bodies": 256,
                "dt": 1e-3, 
                "tsteps": 20000, #20000 
                "save_step":1, #10
                "gamma": 1.0,
                "jx": 0.9,
                "jy": 0.9,
                "jz": 1.0,
                "save_start":0
                },
        gpu_params = {
            "block_size":128, 
            "use_host_mem": True}
                )

    cs.set_platform("GPU")
    cs.run()

    tsteps = cs.params["tsteps"] // cs.params["save_step"]

    time = np.linspace(0, cs.params['tsteps'] * cs.params["dt"], tsteps)
    time_body = np.linspace(0, cs.params['tsteps'] * cs.params["dt"], 
    tsteps*cs.params["num_bodies"])
    
    sx = np.sqrt(3)*np.sin(cs.results[::2]) * np.sin(cs.results[1::2])
    sy = -np.sqrt(3)*np.sin(cs.results[::2]) * np.cos(cs.results[1::2])
    sz = -np.sqrt(3)*np.cos(cs.results[::2])

    sx_mean = np.reshape(sx, (-1, cs.params["num_bodies"])).mean(axis=-1)
    sy_mean = np.reshape(sy, (-1, cs.params["num_bodies"])).mean(axis=-1)
    sz_mean = np.reshape(sz, (-1, cs.params["num_bodies"])).mean(axis=-1)

    plt.figure(figsize=(6,2.5))
    plt.subplot(121)
    plt.title("Intital conf")
    plt.plot(time_body, sx, label="sx")
    plt.plot(time_body, sy, label="sy")
    plt.plot(time_body, sz, label="sz")
    plt.xlabel("t x N", fontsize=14)
    plt.ylabel("s", fontsize=14)
    plt.subplot(122)
    plt.title("Averaged conf")
    plt.plot(time, sx_mean, label="<sx>")
    plt.plot(time, sy_mean, label="<sxy>")
    plt.plot(time, sz_mean, label="<sz>")    
    plt.xlabel("t", fontsize=14)
    plt.ylabel("<s>", fontsize=14)
    plt.legend(frameon=True, prop={"size": 12})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./test_plot_{cusp.cvar.PLATFORM}_hostMem={bool(cs.gpu_params["use_host_mem"])}.png", dpi=200)
    plt.close()

    del cs
    
    return

if __name__ == '__main__':
    main()
