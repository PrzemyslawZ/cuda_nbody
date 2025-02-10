// For extensive commentary see the output version of this code
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cuda_runtime.h>

#define threadsPerBlock 64

__global__ void update(double *arr, double *arr2, size_t n, double h);
cudaError_t cudaCheckError();

/*
 * Return Codes:
 * 0: Success
 * -1: Out of memory on host side
 * -2: Could not open file
 * -3: Could not read CL-argument
 * positive numbers: Cuda error codes
 */
int main(int argc, char *argv[])
{
    size_t length;
    if (argc > 1)
    {
        char *endptr;
        length = strtoul(argv[1], &endptr, 10);

        if (endptr == argv[1])
        {
            puts("Could not read CL-argument");
            return 3;
        }
    }
    else
    {
        length = 999;
    }

    double *arr = (double*)calloc(length, sizeof(double));
    if (arr == NULL)
    {
        puts("Failed to allocate memory for array");
        return 1;
    }
    arr[0] = 1.0;

    double a = 1.0;
    double dx = 1.0;
    double dt = 0.1;
    unsigned int steps = 1000;

    double h = a * dt / (dx * dx);
    double *d_arr = NULL, *d_arr2 = NULL, *tmp;

    // FILE *f = fopen("../var/cu_perf.csv", "a");
    // if (f == NULL)
    // {
    //     puts("Could not open data file");
    //     free(arr);
    //     return 2;
    // }

    cudaMalloc(&d_arr, length * sizeof(double));
    cudaError_t code = cudaCheckError();
    cudaMalloc(&d_arr2, length * sizeof(double));
    cudaError_t code_2 = cudaCheckError();

    if (code != cudaSuccess || code_2 != cudaSuccess)
    {
        if (d_arr != NULL) cudaFree(d_arr);
        if (d_arr2 != NULL) cudaFree(d_arr);
        free(arr);
        return code;
    }

    cudaMemcpy(d_arr, arr, length * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, arr, length * sizeof(double), cudaMemcpyHostToDevice);
    cudaCheckError();

    int threads = 1;
    int blocks = (length + threads - 1) / threads;

    dim3 blockDim(threads);
    dim3 gridDim(blocks);

    struct timeval start, stop;
    double time;

    gettimeofday(&start, NULL);

    for (int i = 0; i < steps; i++)
    {
        update<<<gridDim, blockDim>>>(d_arr, d_arr2, length, h);
        tmp = d_arr;
        d_arr = d_arr2;
        d_arr2 = tmp;
    }

    // cudaMemcpy(arr, d_arr, length * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    gettimeofday(&stop, NULL);

    long seconds  = stop.tv_sec  - start.tv_sec;
    long useconds = stop.tv_usec - start.tv_usec;

    time = seconds * 1e3 + useconds * 1e-3;

    printf("%f\n", time);

    // fclose(f);

    free(arr);
    cudaFree(d_arr);
    cudaFree(d_arr2);

    return 0;
}

__global__ void update(double *arr, double *arr2, size_t n, double h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < n - 1)
    {
        arr2[i] = arr[i] + h * (arr[i + 1] + arr[i - 1] - 2 * arr[i]);
    }
}

cudaError_t cudaCheckError()
{
    cudaError_t code = cudaGetLastError();

    if (code != cudaSuccess)
    {
	    printf("%s: %s\n", cudaGetErrorName(code), cudaGetErrorString(code));
    }

    return code;
}
