#!/bin/bash

# Step 1: Generate .bin files
echo -e "Generating .bin files...\n"
python3 generate_data.py

# Step 2: Compile all needed test_xxx.cu or test_xxx.cpp files
echo -e "\nCompiling test files...\n"
make

# Step 3: Run the generated executables to produce .csv files
echo -e "\nRunning executables to produce .csv files...\n"
for exe in *_test; do
    ./$exe
done

# Step 4: Generate plots
echo -e "\nGenerating plots...\n"
python3 plot_benchmark.py

echo -e "\nAll steps completed!\n"
