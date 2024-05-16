This is a repo used for testing benchmark of sorting algorithms

To test the benchmark of existed algorithms within this repo, just use: ./run.sh

First, the "generate_data.py" will generate 3 kinds of distribution of arrays with different sizes (2^8, 2^11, 2^14, 2^17) and types (int32 and int64)
, storing them in the directory "origin_data"

Second, use make to compile.

Third, run executable files named XXX_test, here XXX represent different algorithms (such as avx2sort and thrust), which will generate csv files of execution time 
of different algorithms and store them in the "result_time" directory.

Fourth, run "plot_benchmark.py", generating plots of benchmark in the directory "result_image"





