#include <iostream>
#include <vector>
#include <string>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h> 
#include "binary_read.h"
#include "measure_sort_time.h"
#include "write_csv.h"

void thrust_sort_int32(std::vector<int32_t>& data) {
    thrust::device_vector<int32_t> d_data = data;  
    cudaDeviceSynchronize(); 
    thrust::sort(d_data.begin(), d_data.end());    
    cudaDeviceSynchronize();
    thrust::copy(d_data.begin(), d_data.end(), data.begin()); 
    cudaDeviceSynchronize(); 
}

void thrust_sort_int64(std::vector<int64_t>& data) {
    thrust::device_vector<int64_t> d_data = data;  
    cudaDeviceSynchronize(); 
    thrust::sort(d_data.begin(), d_data.end());   
    cudaDeviceSynchronize(); 
    thrust::copy(d_data.begin(), d_data.end(), data.begin());  
    cudaDeviceSynchronize(); 
}

int main() {
    int runs = 20;
    std::vector<int> sizes = {8, 11, 14, 17};  // 不同的大小：2^8, 2^11, 2^14, 2^17

    for (int size : sizes) {
        std::vector<int32_t> uniform_data_int32;
        std::vector<int32_t> normal_data_int32;
        std::vector<int32_t> zipf_data_int32;

        std::vector<int64_t> uniform_data_int64;
        std::vector<int64_t> normal_data_int64;
        std::vector<int64_t> zipf_data_int64;

        std::string size_str = std::to_string(1 << size);  // 将大小转换为字符串

        if (!binary_read_file("uniform_data_int32_size_" + size_str + ".bin", uniform_data_int32)) {
            std::cerr << "error opening file: uniform_data_int32_size_" + size_str + ".bin" << std::endl;
            continue;
        }
        if (!binary_read_file("normal_data_int32_size_" + size_str + ".bin", normal_data_int32)) {
            std::cerr << "error opening file: normal_data_int32_size_" + size_str + ".bin" << std::endl;
            continue;
        }
        if (!binary_read_file("zipf_data_int32_size_" + size_str + ".bin", zipf_data_int32)) {
            std::cerr << "error opening file: zipf_data_int32_size_" + size_str + ".bin" << std::endl;
            continue;
        }

        if (!binary_read_file("uniform_data_int64_size_" + size_str + ".bin", uniform_data_int64)) {
            std::cerr << "error opening file: uniform_data_int64_size_" + size_str + ".bin" << std::endl;
            continue;
        }
        if (!binary_read_file("normal_data_int64_size_" + size_str + ".bin", normal_data_int64)) {
            std::cerr << "error opening file: normal_data_int64_size_" + size_str + ".bin" << std::endl;
            continue;
        }
        if (!binary_read_file("zipf_data_int64_size_" + size_str + ".bin", zipf_data_int64)) {
            std::cerr << "error opening file: zipf_data_int64_size_" + size_str + ".bin" << std::endl;
            continue;
        }

        double avg_time_uniform_int32 = measure_sort_time(thrust_sort_int32, uniform_data_int32, runs);
        double avg_time_normal_int32 = measure_sort_time(thrust_sort_int32, normal_data_int32, runs);
        double avg_time_zipf_int32 = measure_sort_time(thrust_sort_int32, zipf_data_int32, runs);

        double avg_time_uniform_int64 = measure_sort_time(thrust_sort_int64, uniform_data_int64, runs);
        double avg_time_normal_int64 = measure_sort_time(thrust_sort_int64, normal_data_int64, runs);
        double avg_time_zipf_int64 = measure_sort_time(thrust_sort_int64, zipf_data_int64, runs);

        std::vector<std::pair<std::string, double>> int32_results = {
            {"Uniform", avg_time_uniform_int32},
            {"Normal", avg_time_normal_int32},
            {"Zipf", avg_time_zipf_int32}
        };
        write_csv("int32_thrust_sort_times_size_" + size_str + ".csv", int32_results);

        std::vector<std::pair<std::string, double>> int64_results = {
            {"Uniform", avg_time_uniform_int64},
            {"Normal", avg_time_normal_int64},
            {"Zipf", avg_time_zipf_int64}
        };
        write_csv("int64_thrust_sort_times_size_" + size_str + ".csv", int64_results);
    }

    return 0;
}
