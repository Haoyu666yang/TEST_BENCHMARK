#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "binary_read.h"
#include "measure_sort_time.h"
#include "write_csv.h"

#define MAX_LEVELS 300

__global__ static void quicksort_kernel(int* values, int N) {
    int pivot, L, R;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int start[MAX_LEVELS];
    int end[MAX_LEVELS];

    start[idx] = idx;
    end[idx] = N - 1;
    while (idx >= 0) {
        L = start[idx];
        R = end[idx];
        if (L < R) {
            pivot = values[L];
            while (L < R) {
                while (values[R] >= pivot && L < R)
                    R--;
                if (L < R)
                    values[L++] = values[R];
                while (values[L] < pivot && L < R)
                    L++;
                if (L < R)
                    values[R--] = values[L];
            }
            values[L] = pivot;
            start[idx + 1] = L + 1;
            end[idx + 1] = end[idx];
            end[idx++] = L;
            if (end[idx] - start[idx] > end[idx - 1] - start[idx - 1]) {
                // swap start[idx] and start[idx-1]
                int tmp = start[idx];
                start[idx] = start[idx - 1];
                start[idx - 1] = tmp;

                // swap end[idx] and end[idx-1]
                tmp = end[idx];
                end[idx] = end[idx - 1];
                end[idx - 1] = tmp;
            }
        }
        else
            idx--;
    }
}

__global__ static void quicksort_kernel64(int64_t* values, int N) {
    int64_t pivot;
    int L, R;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int start[MAX_LEVELS];
    int end[MAX_LEVELS];

    start[idx] = idx;
    end[idx] = N - 1;
    while (idx >= 0) {
        L = start[idx];
        R = end[idx];
        if (L < R) {
            pivot = values[L];
            while (L < R) {
                while (values[R] >= pivot && L < R)
                    R--;
                if (L < R)
                    values[L++] = values[R];
                while (values[L] < pivot && L < R)
                    L++;
                if (L < R)
                    values[R--] = values[L];
            }
            values[L] = pivot;
            start[idx + 1] = L + 1;
            end[idx + 1] = end[idx];
            end[idx++] = L;
            if (end[idx] - start[idx] > end[idx - 1] - start[idx - 1]) {
                // swap start[idx] and start[idx-1]
                int tmp = start[idx];
                start[idx] = start[idx - 1];
                start[idx - 1] = tmp;

                // swap end[idx] and end[idx-1]
                tmp = end[idx];
                end[idx] = end[idx - 1];
                end[idx - 1] = tmp;
            }
        }
        else
            idx--;
    }
}

void quicksort(int* arr, int N, int threadsPerBlock) {
    int* d_arr;
    size_t size = N * sizeof(int);

    cudaMalloc((void**)&d_arr, size);
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    quicksort_kernel<<<blocks, threadsPerBlock>>>(d_arr, N);
    cudaDeviceSynchronize();

    cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

void quicksort64(int64_t* arr, int N, int threadsPerBlock) {
    int64_t* d_arr;
    size_t size = N * sizeof(int64_t);

    cudaMalloc((void**)&d_arr, size);
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    quicksort_kernel64<<<blocks, threadsPerBlock>>>(d_arr, N);
    cudaDeviceSynchronize();

    cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

void quicksortInt32(std::vector<int32_t>& data) {
    quicksort(data.data(), data.size(), 512);
}

void quicksortInt64(std::vector<int64_t>& data) {
    quicksort64(data.data(), data.size(), 512);
}

int main() {
    int runs = 20;
    std::vector<int> sizes = {8, 11, 14, 17};

    for (int size : sizes) {
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

        double avg_time_uniform_int32 = measure_sort_time(quicksortInt32, uniform_data_int32, runs);
        double avg_time_normal_int32 = measure_sort_time(quicksortInt32, normal_data_int32, runs);
        double avg_time_zipf_int32 = measure_sort_time(quicksortInt32, zipf_data_int32, runs);

        double avg_time_uniform_int64 = measure_sort_time(quicksortInt64, uniform_data_int64, runs);
        double avg_time_normal_int64 = measure_sort_time(quicksortInt64, normal_data_int64, runs);
        double avg_time_zipf_int64 = measure_sort_time(quicksortInt64, zipf_data_int64, runs);

        std::vector<std::pair<std::string, double>> int32_results = {
            {"Uniform", avg_time_uniform_int32},
            {"Normal", avg_time_normal_int32},
            {"Zipf", avg_time_zipf_int32}
        };
        write_csv("result_time/int32_quick_sort_times_size_" + size_str + ".csv", int32_results);

        std::vector<std::pair<std::string, double>> int64_results = {
            {"Uniform", avg_time_uniform_int64},
            {"Normal", avg_time_normal_int64},
            {"Zipf", avg_time_zipf_int64}
        };
        write_csv("result_time/int64_quick_sort_times_size_" + size_str + ".csv", int64_results);
    }

    return 0;
}
