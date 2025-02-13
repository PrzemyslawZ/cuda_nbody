from ./cuda_spins.py import CudaSpins


def main():
    cs = CudaSpins()
    cs.get_platform_options()

    cs.set_platform("GPU")
    cs.run()

    print(cs.results)

    del cs
    
    return

if __name__ == '__main__':
    main()
