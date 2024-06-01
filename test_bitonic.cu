#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "binary_read.h"
#include "measure_sort_time.h"
#include "write_csv.h"

__global__ void bitonicSortGPU(int* arr, int j, int k)
{
    unsigned int i, ij;

    i = threadIdx.x + blockDim.x * blockIdx.x;

    ij = i ^ j;

    if (ij > i)
    {
        if ((i & k) == 0)
        {
            if (arr[i] > arr[ij])
            {
                int temp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = temp;
            }
        }
        else
        {
            if (arr[i] < arr[ij])
            {
                int temp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = temp;
            }
        }
    }
}

__global__ void bitonicSortGPU64(int64_t* arr, int j, int k)
{
    unsigned int i, ij;

    i = threadIdx.x + blockDim.x * blockIdx.x;

    ij = i ^ j;

    if (ij > i)
    {
        if ((i & k) == 0)
        {
            if (arr[i] > arr[ij])
            {
                int64_t temp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = temp;
            }
        }
        else
        {
            if (arr[i] < arr[ij])
            {
                int64_t temp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = temp;
            }
        }
    }
}

void bitonicSort(int* arr, int N, int threadsPerBlock)
{
    int* d_arr;
    size_t size = N * sizeof(int);

    cudaMalloc((void**)&d_arr, size);
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    dim3 blocks(N / threadsPerBlock);
    dim3 threads(threadsPerBlock);

    int j, k;
    for (k = 2; k <= N; k <<= 1)
    {
        for (j = k >> 1; j > 0; j = j >> 1)
        {
            bitonicSortGPU <<< blocks, threads >>> (d_arr, j, k);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

void bitonicSort64(int64_t* arr, int N, int threadsPerBlock)
{
    int64_t* d_arr;
    size_t size = N * sizeof(int64_t);

    cudaMalloc((void**)&d_arr, size);
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    dim3 blocks(N / threadsPerBlock);
    dim3 threads(threadsPerBlock);

    int j, k;
    for (k = 2; k <= N; k <<= 1)
    {
        for (j = k >> 1; j > 0; j = j >> 1)
        {
            bitonicSortGPU64 <<< blocks, threads >>> (d_arr, j, k);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

void bitonicSortInt32(std::vector<int32_t>& data)
{
    bitonicSort(data.data(), data.size(), 512);
}

void bitonicSortInt64(std::vector<int64_t>& data)
{
    bitonicSort64(data.data(), data.size(), 512);
}

int main()
{
    int runs = 20;
    std::vector<int> sizes = { 8, 11, 14, 17 };

    for (int size : sizes)
    {
        std::vector<int32_t> uniform_data_int32;
        std::vector<int32_t> normal_data_int32;
        std::vector<int32_t> zipf_data_int32;

        std::vector<int64_t> uniform_data_int64;
        std::vector<int64_t> normal_data_int64;
        std::vector<int64_t> zipf_data_int64;

        std::string size_str = std::to_string(1 << size);

        if (!binary_read_file("origin_data/uniform_data_int32_size_" + size_str + ".bin", uniform_data_int32)) {
            std::cerr << "error opening file: origin_data/uniform_data_int32_size_" + size_str + ".bin" << std::endl;
            continue;
        }
        if (!binary_read_file("origin_data/normal_data_int32_size_" + size_str + ".bin", normal_data_int32)) {
            std::cerr << "error opening file: origin_data/normal_data_int32_size_" + size_str + ".bin" << std::endl;
            continue;
        }
        if (!binary_read_file("origin_data/zipf_data_int32_size_" + size_str + ".bin", zipf_data_int32)) {
            std::cerr << "error opening file: origin_data/zipf_data_int32_size_" + size_str + ".bin" << std::endl;
            continue;
        }

        if (!binary_read_file("origin_data/uniform_data_int64_size_" + size_str + ".bin", uniform_data_int64)) {
            std::cerr << "error opening file: origin_data/uniform_data_int64_size_" + size_str + ".bin" << std::endl;
            continue;
        }
        if (!binary_read_file("origin_data/normal_data_int64_size_" + size_str + ".bin", normal_data_int64)) {
            std::cerr << "error opening file: origin_data/normal_data_int64_size_" + size_str + ".bin" << std::endl;
            continue;
        }
        if (!binary_read_file("origin_data/zipf_data_int64_size_" + size_str + ".bin", zipf_data_int64)) {
            std::cerr << "error opening file: origin_data/zipf_data_int64_size_" + size_str + ".bin" << std::endl;
            continue;
        }

        double avg_time_uniform_int32 = measure_sort_time(bitonicSortInt32, uniform_data_int32, runs);
        double avg_time_normal_int32 = measure_sort_time(bitonicSortInt32, normal_data_int32, runs);
        double avg_time_zipf_int32 = measure_sort_time(bitonicSortInt32, zipf_data_int32, runs);

        double avg_time_uniform_int64 = measure_sort_time(bitonicSortInt64, uniform_data_int64, runs);
        double avg_time_normal_int64 = measure_sort_time(bitonicSortInt64, normal_data_int64, runs);
        double avg_time_zipf_int64 = measure_sort_time(bitonicSortInt64, zipf_data_int64, runs);

        std::vector<std::pair<std::string, double>> int32_results = {
            {"Uniform", avg_time_uniform_int32},
            {"Normal", avg_time_normal_int32},
            {"Zipf", avg_time_zipf_int32}
        };
        write_csv("result_time/int32_bitonic_sort_times_size_" + size_str + ".csv", int32_results);

        std::vector<std::pair<std::string, double>> int64_results = {
            {"Uniform", avg_time_uniform_int64},
            {"Normal", avg_time_normal_int64},
            {"Zipf", avg_time_zipf_int64}
        };
        write_csv("result_time/int64_bitonic_sort_times_size_" + size_str + ".csv", int64_results);
    }

    return 0;
}
