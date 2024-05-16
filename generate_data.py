#!/usr/bin/env python3

import os
import numpy as np

def save_data_to_file(data, filename):
    data.tofile(filename)
    print(f"Saved {filename}")

def generate_uniform_data(size, data_type, seed=None):
    filename = f'uniform_data_{np.dtype(data_type).name}_size_{size}.bin'
    if os.path.exists(filename):
        print(f"File {filename} already exists. Deleting and regenerating.")
        os.remove(filename)
    if seed is not None:
        np.random.seed(seed)
    low = np.iinfo(data_type).min
    high = np.iinfo(data_type).max
    data = np.random.randint(low, high, size, dtype=data_type)
    save_data_to_file(data, filename)

def generate_normal_data(size, data_type, seed=None):
    filename = f'normal_data_{np.dtype(data_type).name}_size_{size}.bin'
    if os.path.exists(filename):
        print(f"File {filename} already exists. Deleting and regenerating.")
        os.remove(filename)
    if seed is not None:
        np.random.seed(seed)
    mean = 0
    std_dev = 1e9 if data_type == np.int64 else 1e6
    data = np.random.normal(mean, std_dev, size).astype(data_type)
    save_data_to_file(data, filename)

def generate_zipf_data(size, data_type, seed=None):
    filename = f'zipf_data_{np.dtype(data_type).name}_size_{size}.bin'
    if os.path.exists(filename):
        print(f"File {filename} already exists. Deleting and regenerating.")
        os.remove(filename)
    if seed is not None:
        np.random.seed(seed)
    a = 2.0
    data = np.random.zipf(a, size).astype(data_type)
    save_data_to_file(data, filename)

seed = 42
sizes = [2**8, 2**11, 2**14, 2**17]

for size in sizes:
    generate_uniform_data(size, np.int32, seed=seed)
    generate_normal_data(size, np.int32, seed=seed)
    generate_zipf_data(size, np.int32, seed=seed)

    generate_uniform_data(size, np.int64, seed=seed)
    generate_normal_data(size, np.int64, seed=seed)
    generate_zipf_data(size, np.int64, seed=seed)
