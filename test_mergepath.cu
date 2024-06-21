#include <iostream>
#include <vector>
#include <string>
#include <curand.h>
#include <cuda_runtime.h>
#include "binary_read.h"
#include "measure_sort_time.h"
#include "write_csv.h"

#define MAX_LEVELS 300

// Helper function for CUDA error checking
void testCUDA(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__))

__global__ void sort_array_2(unsigned int *M, int n) {
    int stride = min(n / 2, 1024) * 2;
    int index = blockIdx.x * blockDim.x * 2;
    for (int i = threadIdx.x * 2; i < blockDim.x * 2; i += stride) {
        if (M[index + i] > M[index + i + 1]) {
            int swap = M[index + i];
            M[index + i] = M[index + i + 1];
            M[index + i + 1] = swap;
        }
    }
}

__global__ void merge_array(unsigned int *m, unsigned int *M, int array_size) {
    int index_array = blockIdx.x * array_size;
    int stride = blockDim.x;
    for (int i = threadIdx.x; i < array_size; i += stride) {
        int size_A = array_size / 2;
        int size_B = array_size / 2;
        int offset;

        int K[2], P[2], Q[2];
        if (i > size_A) {
            K[0] = P[1] = i - size_A;
            K[1] = P[0] = size_A;
        } else {
            K[0] = P[1] = 0;
            K[1] = P[0] = i;
        }
        while (true) {
            offset = abs(K[1] - P[1]) / 2;
            Q[0] = K[0] + offset;
            Q[1] = K[1] - offset;
            if (Q[1] >= 0 && Q[0] <= size_B && (Q[1] == size_A || Q[0] == 0 || M[index_array + Q[1]] > M[index_array + size_A + Q[0] - 1])) {
                if (Q[0] == size_B || Q[1] == 0 || M[index_array + Q[1] - 1] <= M[index_array + size_A + Q[0]]) {
                    if (Q[1] < size_A && (Q[0] == size_B || M[index_array + Q[1]] <= M[index_array + size_A + Q[0]])) {
                        m[index_array + i] = M[index_array + Q[1]];
                    } else {
                        m[index_array + i] = M[index_array + size_A + Q[0]];
                    }
                    break;
                } else {
                    K[0] = Q[0] + 1;
                    K[1] = Q[1] - 1;
                }
            } else {
                P[0] = Q[0] - 1;
                P[1] = Q[1] + 1;
            }
        }
    }
    __syncthreads();
}

void mergepath_sort_int32(std::vector<int32_t>& data) {
    unsigned int *d_M_dev, *d_M_dev_next;
    int n = data.size();
    size_t size = n * sizeof(unsigned int);
    
    testCUDA(cudaMallocManaged(&d_M_dev, size));
    testCUDA(cudaMemcpy(d_M_dev, data.data(), size, cudaMemcpyHostToDevice));
    testCUDA(cudaMallocManaged(&d_M_dev_next, size));

    sort_array_2<<<n / (2 * min(n / 2, 1024)), min(n / 2, 1024)>>>(d_M_dev, n);
    cudaDeviceSynchronize();

    int nb_arrays = n / 2;
    int array_size = 2;
    while (nb_arrays != 1) {
        nb_arrays /= 2;
        array_size *= 2;
        merge_array<<<nb_arrays, min(array_size, 1024)>>>(d_M_dev_next, d_M_dev, array_size);
        std::swap(d_M_dev, d_M_dev_next);
    }

    cudaMemcpy(data.data(), d_M_dev, size, cudaMemcpyDeviceToHost);
    cudaFree(d_M_dev);
    cudaFree(d_M_dev_next);
}

void mergepath_sort_int64(std::vector<int64_t>& data) {
    unsigned int *d_M_dev, *d_M_dev_next;
    int n = data.size();
    size_t size = n * sizeof(unsigned int);
    
    testCUDA(cudaMallocManaged(&d_M_dev, size));
    testCUDA(cudaMemcpy(d_M_dev, data.data(), size, cudaMemcpyHostToDevice));
    testCUDA(cudaMallocManaged(&d_M_dev_next, size));

    sort_array_2<<<n / (2 * min(n / 2, 1024)), min(n / 2, 1024)>>>(d_M_dev, n);
    cudaDeviceSynchronize();

    int nb_arrays = n / 2;
    int array_size = 2;
    while (nb_arrays != 1) {
        nb_arrays /= 2;
        array_size *= 2;
        merge_array<<<nb_arrays, min(array_size, 1024)>>>(d_M_dev_next, d_M_dev, array_size);
        std::swap(d_M_dev, d_M_dev_next);
    }

    cudaMemcpy(data.data(), d_M_dev, size, cudaMemcpyDeviceToHost);
    cudaFree(d_M_dev);
    cudaFree(d_M_dev_next);
}

int main() {
    int runs = 20;
    std::vector<int> sizes = {8, 11, 14, 17, 20};

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

        double avg_time_uniform_int32 = measure_sort_time(mergepath_sort_int32, uniform_data_int32, runs);
        double avg_time_normal_int32 = measure_sort_time(mergepath_sort_int32, normal_data_int32, runs);
        double avg_time_zipf_int32 = measure_sort_time(mergepath_sort_int32, zipf_data_int32, runs);

        double avg_time_uniform_int64 = measure_sort_time(mergepath_sort_int64, uniform_data_int64, runs);
        double avg_time_normal_int64 = measure_sort_time(mergepath_sort_int64, normal_data_int64, runs);
        double avg_time_zipf_int64 = measure_sort_time(mergepath_sort_int64, zipf_data_int64, runs);

        std::vector<std::pair<std::string, double>> int32_results = {
            {"Uniform", avg_time_uniform_int32},
            {"Normal", avg_time_normal_int32},
            {"Zipf", avg_time_zipf_int32}
        };
        write_csv("result_time/int32_mergepath_sort_times_size_" + size_str + ".csv", int32_results);

        std::vector<std::pair<std::string, double>> int64_results = {
            {"Uniform", avg_time_uniform_int64},
            {"Normal", avg_time_normal_int64},
            {"Zipf", avg_time_zipf_int64}
        };
        write_csv("result_time/int64_mergepath_sort_times_size_" + size_str + ".csv", int64_results);
    }

    return 0;
}
