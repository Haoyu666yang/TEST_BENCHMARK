#include <iostream>
#include <vector>
#include <string>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include "binary_read.h"
#include "measure_sort_time.h"
#include "write_csv.h"

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(x)); \
    exit(EXIT_FAILURE);}} while(0)

template <typename T>
void cub_sort(std::vector<T>& data) {
    T *d_input, *d_output;
    size_t num_items = data.size();
    CUDA_CALL(cudaMalloc(&d_input, num_items * sizeof(T)));
    CUDA_CALL(cudaMalloc(&d_output, num_items * sizeof(T)));
    CUDA_CALL(cudaMemcpy(d_input, data.data(), num_items * sizeof(T), cudaMemcpyHostToDevice));

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CUDA_CALL(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, num_items));
    CUDA_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CUDA_CALL(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, num_items));

    CUDA_CALL(cudaMemcpy(data.data(), d_output, num_items * sizeof(T), cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_input));
    CUDA_CALL(cudaFree(d_output));
    CUDA_CALL(cudaFree(d_temp_storage));
}

void cub_sort_int32(std::vector<int32_t>& data) {
    cub_sort(data);
}

void cub_sort_int64(std::vector<int64_t>& data) {
    cub_sort(data);
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

        double avg_time_uniform_int32 = measure_sort_time(cub_sort_int32, uniform_data_int32, runs);
        double avg_time_normal_int32 = measure_sort_time(cub_sort_int32, normal_data_int32, runs);
        double avg_time_zipf_int32 = measure_sort_time(cub_sort_int32, zipf_data_int32, runs);

        double avg_time_uniform_int64 = measure_sort_time(cub_sort_int64, uniform_data_int64, runs);
        double avg_time_normal_int64 = measure_sort_time(cub_sort_int64, normal_data_int64, runs);
        double avg_time_zipf_int64 = measure_sort_time(cub_sort_int64, zipf_data_int64, runs);

        std::vector<std::pair<std::string, double>> int32_results = {
            {"Uniform", avg_time_uniform_int32},
            {"Normal", avg_time_normal_int32},
            {"Zipf", avg_time_zipf_int32}
        };
        write_csv("result_time/int32_cub_sort_times_size_" + size_str + ".csv", int32_results);

        std::vector<std::pair<std::string, double>> int64_results = {
            {"Uniform", avg_time_uniform_int64},
            {"Normal", avg_time_normal_int64},
            {"Zipf", avg_time_zipf_int64}
        };
        write_csv("result_time/int64_cub_sort_times_size_" + size_str + ".csv", int64_results);
    }

    return 0;
}


// #include <iostream>
// #include <vector>
// #include <string>
// #include <thrust/sort.h>
// #include <thrust/device_vector.h>
// #include <cuda_runtime.h> 
// #include "binary_read.h"
// #include "measure_sort_time.h"
// #include "write_csv.h"

// void thrust_sort_int32(std::vector<int32_t>& data) {
//     thrust::device_vector<int32_t> d_data = data;  
//     cudaDeviceSynchronize(); 
//     thrust::sort(d_data.begin(), d_data.end());    
//     cudaDeviceSynchronize();
//     thrust::copy(d_data.begin(), d_data.end(), data.begin()); 
//     cudaDeviceSynchronize(); 
// }

// void thrust_sort_int64(std::vector<int64_t>& data) {
//     thrust::device_vector<int64_t> d_data = data;  
//     cudaDeviceSynchronize(); 
//     thrust::sort(d_data.begin(), d_data.end());   
//     cudaDeviceSynchronize(); 
//     thrust::copy(d_data.begin(), d_data.end(), data.begin());  
//     cudaDeviceSynchronize(); 
// }

// int main() {
//     int runs = 20;
//     std::vector<int> sizes = {8, 11, 14, 17};  

//     for (int size : sizes) {
//         std::vector<int32_t> uniform_data_int32;
//         std::vector<int32_t> normal_data_int32;
//         std::vector<int32_t> zipf_data_int32;

//         std::vector<int64_t> uniform_data_int64;
//         std::vector<int64_t> normal_data_int64;
//         std::vector<int64_t> zipf_data_int64;

//         std::string size_str = std::to_string(1 << size);  

//         if (!binary_read_file("origin_data/uniform_data_int32_size_" + size_str + ".bin", uniform_data_int32)) {
//             std::cerr << "error opening file: origin_data/uniform_data_int32_size_" + size_str + ".bin" << std::endl;
//             continue;
//         }
//         if (!binary_read_file("origin_data/normal_data_int32_size_" + size_str + ".bin", normal_data_int32)) {
//             std::cerr << "error opening file: origin_data/normal_data_int32_size_" + size_str + ".bin" << std::endl;
//             continue;
//         }
//         if (!binary_read_file("origin_data/zipf_data_int32_size_" + size_str + ".bin", zipf_data_int32)) {
//             std::cerr << "error opening file: origin_data/zipf_data_int32_size_" + size_str + ".bin" << std::endl;
//             continue;
//         }

//         if (!binary_read_file("origin_data/uniform_data_int64_size_" + size_str + ".bin", uniform_data_int64)) {
//             std::cerr << "error opening file: origin_data/uniform_data_int64_size_" + size_str + ".bin" << std::endl;
//             continue;
//         }
//         if (!binary_read_file("origin_data/normal_data_int64_size_" + size_str + ".bin", normal_data_int64)) {
//             std::cerr << "error opening file: origin_data/normal_data_int64_size_" + size_str + ".bin" << std::endl;
//             continue;
//         }
//         if (!binary_read_file("origin_data/zipf_data_int64_size_" + size_str + ".bin", zipf_data_int64)) {
//             std::cerr << "error opening file: origin_data/zipf_data_int64_size_" + size_str + ".bin" << std::endl;
//             continue;
//         }

//         double avg_time_uniform_int32 = measure_sort_time(thrust_sort_int32, uniform_data_int32, runs);
//         double avg_time_normal_int32 = measure_sort_time(thrust_sort_int32, normal_data_int32, runs);
//         double avg_time_zipf_int32 = measure_sort_time(thrust_sort_int32, zipf_data_int32, runs);

//         double avg_time_uniform_int64 = measure_sort_time(thrust_sort_int64, uniform_data_int64, runs);
//         double avg_time_normal_int64 = measure_sort_time(thrust_sort_int64, normal_data_int64, runs);
//         double avg_time_zipf_int64 = measure_sort_time(thrust_sort_int64, zipf_data_int64, runs);

//         std::vector<std::pair<std::string, double>> int32_results = {
//             {"Uniform", avg_time_uniform_int32},
//             {"Normal", avg_time_normal_int32},
//             {"Zipf", avg_time_zipf_int32}
//         };
//         write_csv("result_time/int32_thrust_sort_times_size_" + size_str + ".csv", int32_results);

//         std::vector<std::pair<std::string, double>> int64_results = {
//             {"Uniform", avg_time_uniform_int64},
//             {"Normal", avg_time_normal_int64},
//             {"Zipf", avg_time_zipf_int64}
//         };
//         write_csv("result_time/int64_thrust_sort_times_size_" + size_str + ".csv", int64_results);

//     }

//     return 0;
// }
