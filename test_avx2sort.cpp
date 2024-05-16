#include <iostream>
#include <vector>
#include <string>
#include <immintrin.h>
#include <algorithm>
#include "binary_read.h"
#include "measure_sort_time.h"
#include "write_csv.h"

// AVX2 optimized partition function for int32_t
int partition_avx2(std::vector<int32_t>& arr, int low, int high) {
    int32_t pivot = arr[high];
    int i = low - 1;
    int j = low;

    for (; j <= high - 7; j += 8) {
        __m256i pivot_vec = _mm256_set1_epi32(pivot);
        __m256i data_vec = _mm256_loadu_si256((__m256i*)&arr[j]);

        __m256i mask = _mm256_cmpgt_epi32(pivot_vec, data_vec);
        int mask_bits = _mm256_movemask_epi8(mask);

        for (int k = 0; k < 8; ++k) {
            if (mask_bits & (1 << (k * 4))) {
                ++i;
                std::swap(arr[i], arr[j + k]);
            }
        }
    }

    for (; j <= high - 1; ++j) {
        if (arr[j] < pivot) {
            ++i;
            std::swap(arr[i], arr[j]);
        }
    }

    std::swap(arr[i + 1], arr[high]);
    return (i + 1);
}

// AVX2 optimized quicksort function for int32_t
void quicksort_avx2(std::vector<int32_t>& arr, int low, int high) {
    if (low < high) {
        int pi = partition_avx2(arr, low, high);
        quicksort_avx2(arr, low, pi - 1);
        quicksort_avx2(arr, pi + 1, high);
    }
}

void avx2_sort(std::vector<int32_t>& arr) {
    quicksort_avx2(arr, 0, arr.size() - 1);
}

// AVX2 optimized partition function for int64_t
int partition_avx2_64(std::vector<int64_t>& arr, int low, int high) {
    int64_t pivot = arr[high];
    int i = low - 1;
    int j = low;

    for (; j <= high - 3; j += 4) {
        __m256i pivot_vec = _mm256_set1_epi64x(pivot);
        __m256i data_vec = _mm256_loadu_si256((__m256i*)&arr[j]);

        __m256i mask = _mm256_cmpgt_epi64(pivot_vec, data_vec);
        int mask_bits = _mm256_movemask_epi8(mask);

        for (int k = 0; k < 4; ++k) {
            if (mask_bits & (1 << (k * 8))) {
                ++i;
                std::swap(arr[i], arr[j + k]);
            }
        }
    }

    for (; j <= high - 1; ++j) {
        if (arr[j] < pivot) {
            ++i;
            std::swap(arr[i], arr[j]);
        }
    }

    std::swap(arr[i + 1], arr[high]);
    return (i + 1);
}

// AVX2 optimized quicksort function for int64_t
void quicksort_avx2_64(std::vector<int64_t>& arr, int low, int high) {
    if (low < high) {
        int pi = partition_avx2_64(arr, low, high);
        quicksort_avx2_64(arr, low, pi - 1);
        quicksort_avx2_64(arr, pi + 1, high);
    }
}

void avx2_sort(std::vector<int64_t>& arr) {
    quicksort_avx2_64(arr, 0, arr.size() - 1);
}

void print_data(const std::vector<int32_t>& data, const std::string& label) {
    std::cout << label << ": ";
    for (size_t i = 0; i < 10 && i < data.size(); ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

void print_data(const std::vector<int64_t>& data, const std::string& label) {
    std::cout << label << ": ";
    for (size_t i = 0; i < 10 && i < data.size(); ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

bool is_sorted(const std::vector<int32_t>& arr) {
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i - 1] > arr[i]) {
            return false;
        }
    }
    return true;
}

bool is_sorted(const std::vector<int64_t>& arr) {
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i - 1] > arr[i]) {
            return false;
        }
    }
    return true;
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

        if (!binary_read_file("uniform_data_int32_size_" + size_str + ".bin", uniform_data_int32)) {
            continue;
        }
        if (!binary_read_file("normal_data_int32_size_" + size_str + ".bin", normal_data_int32)) {
            continue;
        }
        if (!binary_read_file("zipf_data_int32_size_" + size_str + ".bin", zipf_data_int32)) {
            continue;
        }

        if (!binary_read_file("uniform_data_int64_size_" + size_str + ".bin", uniform_data_int64)) {
            continue;
        }
        if (!binary_read_file("normal_data_int64_size_" + size_str + ".bin", normal_data_int64)) {
            continue;
        }
        if (!binary_read_file("zipf_data_int64_size_" + size_str + ".bin", zipf_data_int64)) {
            continue;
        }

        double avg_time_uniform_int32 = measure_sort_time(avx2_sort, uniform_data_int32, runs);
        double avg_time_normal_int32 = measure_sort_time(avx2_sort, normal_data_int32, runs);
        double avg_time_zipf_int32 = measure_sort_time(avx2_sort, zipf_data_int32, runs);

        double avg_time_uniform_int64 = measure_sort_time(avx2_sort, uniform_data_int64, runs);
        double avg_time_normal_int64 = measure_sort_time(avx2_sort, normal_data_int64, runs);
        double avg_time_zipf_int64 = measure_sort_time(avx2_sort, zipf_data_int64, runs);

        std::vector<std::pair<std::string, double>> int32_results = {
            {"Uniform", avg_time_uniform_int32},
            {"Normal", avg_time_normal_int32},
            {"Zipf", avg_time_zipf_int32}
        };
        write_csv("int32_avx2sort_times_size_" + size_str + ".csv", int32_results);

        std::vector<std::pair<std::string, double>> int64_results = {
            {"Uniform", avg_time_uniform_int64},
            {"Normal", avg_time_normal_int64},
            {"Zipf", avg_time_zipf_int64}
        };
        write_csv("int64_avx2sort_times_size_" + size_str + ".csv", int64_results);
    }

    return 0;
}
