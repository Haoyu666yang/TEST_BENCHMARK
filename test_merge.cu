#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "binary_read.h"
#include "measure_sort_time.h"
#include "write_csv.h"

// Merge function for int32
__device__ void Merge(int *arr, int *temp, int left, int middle, int right)
{
    int i = left;
    int j = middle;
    int k = left;

    while (i < middle && j < right)
    {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i < middle)
        temp[k++] = arr[i++];
    while (j < right)
        temp[k++] = arr[j++];

    for (int x = left; x < right; x++)
        arr[x] = temp[x];
}

// Merge function for int64
__device__ void Merge(int64_t *arr, int64_t *temp, int left, int middle, int right)
{
    int i = left;
    int j = middle;
    int k = left;

    while (i < middle && j < right)
    {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i < middle)
        temp[k++] = arr[i++];
    while (j < right)
        temp[k++] = arr[j++];

    for (int x = left; x < right; x++)
        arr[x] = temp[x];
}

// GPU Kernel for Merge Sort for int32
__global__ void MergeSortGPU(int *arr, int *temp, int n, int width)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int left = tid * width;
    int middle = left + width / 2;
    int right = left + width;

    if (left < n && middle < n)
    {
        Merge(arr, temp, left, middle, right);
    }
}

// GPU Kernel for Merge Sort for int64
__global__ void MergeSortGPU(int64_t *arr, int64_t *temp, int n, int width)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int left = tid * width;
    int middle = left + width / 2;
    int right = left + width;

    if (left < n && middle < n)
    {
        Merge(arr, temp, left, middle, right);
    }
}

// Host function for Merge Sort for int32
void merge_sort_int32(std::vector<int32_t> &data)
{
    int n = data.size();
    int *d_arr;
    int *d_temp;
    cudaMalloc(&d_arr, n * sizeof(int32_t));
    cudaMalloc(&d_temp, n * sizeof(int32_t));

    cudaMemcpy(d_arr, data.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice);

    for (int width = 2; width / 2 < n; width *= 2)
    {
        int num_blocks = (n + width - 1) / width;
        MergeSortGPU<<<num_blocks, 1>>>(d_arr, d_temp, n, width);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(data.data(), d_arr, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_temp);
    if(!std::is_sorted(data.begin(), data.end()))
    {
        std::cerr << "Error: merge_sort_int32 failed" << std::endl;
    }
    
}

// Host function for Merge Sort for int64
void merge_sort_int64(std::vector<int64_t> &data)
{
    int n = data.size();
    int64_t *d_arr;
    int64_t *d_temp;
    cudaMalloc(&d_arr, n * sizeof(int64_t));
    cudaMalloc(&d_temp, n * sizeof(int64_t));

    cudaMemcpy(d_arr, data.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice);

    for (int width = 2; width / 2 < n; width *= 2)
    {
        int num_blocks = (n + width - 1) / width;
        MergeSortGPU<<<num_blocks, 1>>>(d_arr, d_temp, n, width);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(data.data(), d_arr, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_temp);
        if(!std::is_sorted(data.begin(), data.end()))
    {
        std::cerr << "Error: merge_sort_int32 failed" << std::endl;
    }
}

int main()
{
    int runs = 20;
    std::vector<int> sizes = {8, 11, 14, 17, 20};

    for (int size : sizes)
    {
        std::vector<int32_t> uniform_data_int32;
        std::vector<int32_t> normal_data_int32;
        std::vector<int32_t> zipf_data_int32;

        std::vector<int64_t> uniform_data_int64;
        std::vector<int64_t> normal_data_int64;
        std::vector<int64_t> zipf_data_int64;

        std::string size_str = std::to_string(1 << size);

        if (!binary_read_file("origin_data/uniform_data_int32_size_" + size_str + ".bin", uniform_data_int32))
        {
            std::cerr << "error opening file: origin_data/uniform_data_int32_size_" + size_str + ".bin" << std::endl;
            continue;
        }
        if (!binary_read_file("origin_data/normal_data_int32_size_" + size_str + ".bin", normal_data_int32))
        {
            std::cerr << "error opening file: origin_data/normal_data_int32_size_" + size_str + ".bin" << std::endl;
            continue;
        }
        if (!binary_read_file("origin_data/zipf_data_int32_size_" + size_str + ".bin", zipf_data_int32))
        {
            std::cerr << "error opening file: origin_data/zipf_data_int32_size_" + size_str + ".bin" << std::endl;
            continue;
        }

        if (!binary_read_file("origin_data/uniform_data_int64_size_" + size_str + ".bin", uniform_data_int64))
        {
            std::cerr << "error opening file: origin_data/uniform_data_int64_size_" + size_str + ".bin" << std::endl;
            continue;
        }
        if (!binary_read_file("origin_data/normal_data_int64_size_" + size_str + ".bin", normal_data_int64))
        {
            std::cerr << "error opening file: origin_data/normal_data_int64_size_" + size_str + ".bin" << std::endl;
            continue;
        }
        if (!binary_read_file("origin_data/zipf_data_int64_size_" + size_str + ".bin", zipf_data_int64))
        {
            std::cerr << "error opening file: origin_data/zipf_data_int64_size_" + size_str + ".bin" << std::endl;
            continue;
        }

        double avg_time_uniform_int32 = measure_sort_time(merge_sort_int32, uniform_data_int32, runs);
        double avg_time_normal_int32 = measure_sort_time(merge_sort_int32, normal_data_int32, runs);
        double avg_time_zipf_int32 = measure_sort_time(merge_sort_int32, zipf_data_int32, runs);

        double avg_time_uniform_int64 = measure_sort_time(merge_sort_int64, uniform_data_int64, runs);
        double avg_time_normal_int64 = measure_sort_time(merge_sort_int64, normal_data_int64, runs);
        double avg_time_zipf_int64 = measure_sort_time(merge_sort_int64, zipf_data_int64, runs);

        std::vector<std::pair<std::string, double>> int32_results = {
            {"Uniform", avg_time_uniform_int32},
            {"Normal", avg_time_normal_int32},
            {"Zipf", avg_time_zipf_int32}};
        write_csv("result_time/int32_merge_sort_times_size_" + size_str + ".csv", int32_results);

        std::vector<std::pair<std::string, double>> int64_results = {
            {"Uniform", avg_time_uniform_int64},
            {"Normal", avg_time_normal_int64},
            {"Zipf", avg_time_zipf_int64}};
        write_csv("result_time/int64_merge_sort_times_size_" + size_str + ".csv", int64_results);
    }

    return 0;
}
