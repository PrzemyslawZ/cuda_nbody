from pycu_spins import CudaSpins, np, cusp
import matplotlib.pyplot as plt

def main():
    
    cs = CudaSpins(
        params = 
                {
                "system_type": 16,
                "num_bodies": 1024,
                "dt": 1e-3, #keep it 
                "tsteps": 200, #20000, #keep it 
                "save_step":1,#10, #keep it 
                "nx_spins": 64,
                "ny_spins": 64,
                "nz_spins": 1,
                "gamma1": 0,
                "gamma2": 0,
                "nth1": 0,
                "nth2": 0,
                "gamma": 1.0,
                "gamma_phase": 0,
                "jx": 0.9,
                "jy": 0.9,
                "jz": 1.0,
                "Omega_x": 0,
                "Omega_y": 0,
                "Omega_z": 0,
                "Omega_D": 0,
                "save_start":0
                },
        gpu_params = {
            "block_size":256, 
            "blocks_num": 1,
            "use_host_mem": True}
                )
    # cs.get_platform_options()

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

    # print(cs.results)

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
