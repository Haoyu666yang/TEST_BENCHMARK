import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculate_throughput(size, time_seconds, element_size_bytes):
    data_size_bytes = size * element_size_bytes
    data_size_gb = data_size_bytes *8 / (10 ** 9)  # GB
    throughput_gbps = data_size_gb / time_seconds  # GB/s
    return throughput_gbps

sizes = [2**8, 2**11, 2**14, 2**17]
element_sizes = {'int32': 4, 'int64': 8}
distributions = ['Uniform', 'Normal', 'Zipf']

algorithms = ['avx2sort', 'thrust_sort', 'cub_sort']  # may need to change

x_labels = [f'{int(np.log2(size))}' for size in sizes]
x_ticks = [8, 11, 14, 17]

# Create the result_image directory if it doesn't exist
result_dir = 'result_image'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

for distribution in distributions:
    for data_type in ['int32', 'int64']:
        throughput_data = {alg: [] for alg in algorithms}
        
        for size in sizes:
            size_str = str(size)
            for algorithm in algorithms:
                filename = os.path.join('result_time', f'{data_type}_{algorithm}_times_size_{size_str}.csv')
                print(f'Reading file: {filename}') 

                if not os.path.exists(filename):
                    print(f"File not found: {filename}")
                    continue

                df = pd.read_csv(filename)
                time = df[df['Distribution'] == distribution]['Average Time (s)'].values[0]
                throughput = calculate_throughput(size, time, element_sizes[data_type])
                throughput_data[algorithm].append(throughput)

        output_filename = os.path.join(result_dir, f'{distribution}_{data_type}_throughput.png')
        if os.path.exists(output_filename):
            os.remove(output_filename)

        plt.figure(figsize=(10, 6))
        for algorithm in algorithms:
            plt.plot(x_ticks, throughput_data[algorithm], marker='o', linestyle='-', label=f'{algorithm}')

        plt.xlabel('Input Size (2^x)')
        plt.ylabel('Throughput (GB/s)')
        plt.title(f'Throughput for {distribution} Distribution ({data_type})')
        plt.xticks(x_ticks)  
        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.show()
