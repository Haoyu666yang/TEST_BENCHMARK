#ifndef MEASURE_SORT_TIME_H
#define MEASURE_SORT_TIME_H

#include <chrono>
#include <sys/time.h>
#include <vector>

double my_clock();

template <typename T>
double measure_sort_time(void (*sort_func)(std::vector<T>&), std::vector<T>& data, int runs);

#endif