This is a repo used for testing benchmarks of sorting algorithms

To test the benchmark of existing algorithms within this repo, just use: ./run.sh

First, the "generate_data.py" will generate 3 kinds of distribution of arrays with different sizes (2^8, 2^11, 2^14, 2^17) and types (int32 and int64)
, storing them in the directory "origin_data"

Second, use make to compile, generating executable files named XXX_test, here XXX represents different algorithms (such as avx2sort and thrust)

Third, run executable files named XXX_test, which will generate CSV files of execution time 
of different algorithms and store them in the "result_time" directory.

Fourth, run "plot_benchmark.py", generating plots of benchmark in the directory "result_image"

___________________________________________________________________________________________________________________________
binary_read.cpp, measure_sort_time.cpp, write_csv.cpp are helper functions used in test_xxx.cpp or test_xxx.cu, latter ones are algorithms we'd like to
test the benchmarks






