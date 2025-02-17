from pycu_spins import CudaSpins, cusp

# import _cuda_interface as cusp

def main():
    
    cs = CudaSpins()
    cs.get_platform_options()

    cs.set_platform("GPU")
    cs.run()

    print(cs.results)
    # # results = cusp.simulate(input_buffer, {"a":100}, {})
    # # for i in range(10000):        # Set some values
    # #     input_buffer = i
    # print(input_buffer[0])
    # del input_buffer
    del cs
    
    return

if __name__ == '__main__':
    main()
