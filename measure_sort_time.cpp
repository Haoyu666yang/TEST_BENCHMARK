#include "measure_sort_time.h"

double my_clock()
{
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

template <typename T>
double measure_sort_time(void (*sort_func)(std::vector<T>&), std::vector<T>& data, int runs) {
    double start_time = my_clock();
    for (int i = 0; i < runs; ++i) {
        std::vector<T> data_copy = data; // Make a copy of the data for each run
        sort_func(data_copy); // Sort the data
    }
    double end_time = my_clock();
    return (end_time - start_time) / runs;
}

// Explicit template instantiation
template double measure_sort_time<int32_t>(void (*sort_func)(std::vector<int32_t>&), std::vector<int32_t>& data, int runs);
template double measure_sort_time<int64_t>(void (*sort_func)(std::vector<int64_t>&), std::vector<int64_t>& data, int runs);