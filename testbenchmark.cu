#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <chrono>
#include <sys/time.h>
#include <fstream> 
#include <algorithm>
#include <vector>



#define TOPK 10
#define BITSHIFT 21

double my_clock()
{
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main()
{
    std::ofstream outfile("benchmark_results.csv");

    int runs = 1;
    // Generate 32M random numbers serially.
    thrust::default_random_engine rng(1337);


    thrust::uniform_int_distribution<int> dist1(-10000, 10000);        // distribution 1
    thrust::normal_distribution<float> dist2(-10000.0, 10000.0);       // distribution 2
    thrust::uniform_real_distribution<float> dist3(-10000.0, 10000.0); // distribution 3

    thrust::host_vector<int> h_vec1(32 << BITSHIFT);
    thrust::generate(h_vec1.begin(), h_vec1.end(), [&]
                     { return dist1(rng); });
    
    thrust::host_vector<float> h_vec2(32 << BITSHIFT);
    thrust::generate(h_vec2.begin(), h_vec2.end(), [&]
                     { return dist2(rng); });

    thrust::host_vector<float> h_vec3(32 << BITSHIFT);
    thrust::generate(h_vec3.begin(), h_vec3.end(), [&]
                     { return dist3(rng); });


    std::vector<int> vec1(h_vec1.size());
    thrust::copy(h_vec1.begin(), h_vec1.end(), vec1.begin());

    std::vector<float> vec2(h_vec2.size());
    thrust::copy(h_vec2.begin(), h_vec2.end(), vec2.begin());

    std::vector<float> vec3(h_vec3.size());
    thrust::copy(h_vec3.begin(), h_vec3.end(), vec3.begin());

    thrust::device_vector<int> d_vec1 = h_vec1;
    thrust::device_vector<float> d_vec2 = h_vec2;
    thrust::device_vector<float> d_vec3 = h_vec3; 

    outfile << "DataType, AverageDuration(seconds)\n";

// distribution1
    // Transfer data to the device.
    std::cout<< "before sort d_vec1..."  <<std::endl;
    for(int i = 0; i < TOPK; ++i) 
    {
        std::cout << d_vec1[i] << " ";
    }
    std::cout << std::endl;
    cudaDeviceSynchronize();
    double start_time1 = my_clock();
    for (int i = 0; i < runs; i++)
    {
        // Sort data on the device.
        thrust::sort(d_vec1.begin(), d_vec1.end());

        // Transfer data back to host.
        // thrust::copy(d_vec1.begin(), d_vec1.end(), h_vec1.begin());
    }
    cudaDeviceSynchronize();
    double end_time1 = my_clock();
    double averageDuration1 = (end_time1 - start_time1) / runs;
    std::cout<< averageDuration1 << "s" <<std::endl;
    outfile << "DEsort_Integer, " << averageDuration1 << "\n";
    std::cout<< "after sort d_vec1..."  <<std::endl;
    for(int i = 0; i < TOPK; ++i) 
    {
        std::cout << d_vec1[i] << " ";
    }
    std::cout << std::endl;

    std::cout<< "before sort vec1..."  <<std::endl;
    for(int i = 0; i < TOPK; ++i) 
    {
        std::cout << vec1[i] << " ";
    }
    std::cout << std::endl;
    start_time1 = my_clock();
    for (int i = 0; i < runs; i++)
    {
        std::sort(vec1.begin(),vec1.end());
    }
    end_time1 = my_clock();
    averageDuration1 = (end_time1 - start_time1) / runs;
    std::cout<< averageDuration1 << "s" <<std::endl;
    outfile << "STLsort_Integer, " << averageDuration1 << "\n";
    std::cout<< "after sort vec1..."  <<std::endl;
    for(int i = 0; i < TOPK; ++i) 
    {
        std::cout << vec1[i] << " ";
    }
    std::cout << std::endl;

    std::cout<< "before sort h_vec1..."  <<std::endl;
    for(int i = 0; i < TOPK; ++i) 
    {
        std::cout << h_vec1[i] << " ";
    }
    std::cout << std::endl;
    start_time1 = my_clock();
    cudaDeviceSynchronize();
    for (int i = 0; i < runs; i++)
    {
        thrust::sort(h_vec1.begin(), h_vec1.end());
    }
    cudaDeviceSynchronize();
    end_time1 = my_clock();
    averageDuration1 = (end_time1 - start_time1) / runs;
    std::cout<< averageDuration1 << "s" <<std::endl;
    outfile << "HOsort_Integer, " << averageDuration1 << "\n";
    std::cout<< "after sort h_vec1..."  <<std::endl;
    for(int i = 0; i < TOPK; ++i) 
    {
        std::cout << h_vec1[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;




// distribution2

    std::cout<< "before sort d_vec2..."  <<std::endl;
    for(int i = 0; i < TOPK; ++i) 
    {
        std::cout << d_vec2[i] << " ";
    }
    std::cout << std::endl;
    cudaDeviceSynchronize();
    double start_time2 = my_clock();
    for (int i = 0; i < runs; i++)
    {
        // Sort data on the device.
        thrust::sort(d_vec2.begin(), d_vec2.end());

        // Transfer data back to host.
        // thrust::copy(d_vec2.begin(), d_vec2.end(), h_vec2.begin());
    }
        cudaDeviceSynchronize();
    double end_time2 = my_clock();
    double averageDuration2 = (end_time2 - start_time2) / runs;
    std::cout<< averageDuration2 << "s" <<std::endl;
    outfile << "DEsort_Normal Float, " << averageDuration2 << "\n";
    std::cout<< "after sort d_vec2..."  <<std::endl;
    for(int i = 0; i < TOPK; ++i) 
    {
        std::cout << d_vec2[i] << " ";
    }
    std::cout << std::endl;

    std::cout<< "before sort vec2..."  <<std::endl;
    for(int i = 0; i < TOPK; ++i) 
    {
        std::cout << vec2[i] << " ";
    }
    std::cout << std::endl;
    start_time1 = my_clock();
    for (int i = 0; i < runs; i++)
    {
        std::sort(vec2.begin(),vec2.end());
    }
    end_time1 = my_clock();
    averageDuration1 = (end_time1 - start_time1) / runs;
    std::cout<< averageDuration1 << "s" <<std::endl;
    outfile << "STLsort_Normal, " << averageDuration1 << "\n";
    std::cout<< "after sort vec2..."  <<std::endl;
    for(int i = 0; i < TOPK; ++i) 
    {
        std::cout << vec2[i] << " ";
    }
    std::cout << std::endl;

    std::cout<< "before sort h_vec2..."  <<std::endl;
    for(int i = 0; i < TOPK; ++i) 
    {
        std::cout << h_vec2[i] << " ";
    }
    std::cout << std::endl;
    start_time1 = my_clock();
    for (int i = 0; i < runs; i++)
    {
        thrust::sort(h_vec2.begin(), h_vec2.end());
    } 
    end_time1 = my_clock();
    averageDuration1 = (end_time1 - start_time1) / runs;
    std::cout<< averageDuration1 << "s" <<std::endl;
    outfile << "HOsort_Normal, " << averageDuration1 << "\n";
    std::cout<< "after sort h_vec2..."  <<std::endl;
    for(int i = 0; i < TOPK; ++i) 
    {
        std::cout << h_vec2[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;


// distribution3

    std::cout<< "before sort d_vec3..."  <<std::endl;
    for(int i = 0; i < TOPK; ++i) 
    {
        std::cout << d_vec3[i] << " ";
    }
    std::cout << std::endl;
    cudaDeviceSynchronize();
    double start_time3 = my_clock();
    for (int i = 0; i < runs; i++)
    {
        // Sort data on the device.
        thrust::sort(d_vec3.begin(), d_vec3.end());

        // Transfer data back to host.
        // thrust::copy(d_vec3.begin(), d_vec3.end(), h_vec3.begin());
    }
        cudaDeviceSynchronize();
    double end_time3 = my_clock();
    double averageDuration3 = (end_time3 - start_time3) / runs;
    std::cout<< averageDuration3 << "s" <<std::endl;
    outfile << "DEsort_Uniform Float, " << averageDuration3 << "\n";
    std::cout<< "after sort d_vec3..."  <<std::endl;
    for(int i = 0; i < TOPK; ++i) 
    {
        std::cout << d_vec3[i] << " ";
    }
    std::cout << std::endl;

    std::cout<< "before sort vec3..."  <<std::endl;
    for(int i = 0; i < TOPK; ++i) 
    {
        std::cout << vec3[i] << " ";
    }
    std::cout << std::endl;
    start_time1 = my_clock();
    for (int i = 0; i < runs; i++)
    {
        std::sort(vec3.begin(),vec3.end());
    }
    end_time1 = my_clock();
    averageDuration1 = (end_time1 - start_time1) / runs;
    std::cout<< averageDuration1 << "s" <<std::endl;
    outfile << "STLsort_Uniform Float, " << averageDuration1 << "\n";
    std::cout<< "after sort vec3..."  <<std::endl;
    for(int i = 0; i < TOPK; ++i) 
    {
        std::cout << vec3[i] << " ";
    }
    std::cout << std::endl;

    std::cout<< "before sort h_vec3..."  <<std::endl;
    for(int i = 0; i < TOPK; ++i) 
    {
        std::cout << h_vec3[i] << " ";
    }
    std::cout << std::endl;
    start_time1 = my_clock();
    for (int i = 0; i < runs; i++)
    {
        thrust::sort(h_vec3.begin(), h_vec3.end());
    }
    end_time1 = my_clock();
    averageDuration1 = (end_time1 - start_time1) / runs;
    std::cout<< averageDuration1 << "s" <<std::endl;
    outfile << "HOsort_Uniform Float, " << averageDuration1 << "\n";
    std::cout<< "after sort h_vec3..."  <<std::endl;
    for(int i = 0; i < TOPK; ++i) 
    {
        std::cout << h_vec3[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;


    
    outfile.close();
}



