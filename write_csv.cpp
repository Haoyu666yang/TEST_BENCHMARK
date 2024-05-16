#include "write_csv.h"
#include <fstream>
#include <iostream>
#include <cstdio> // For std::remove

void write_csv(const std::string& filename, const std::vector<std::pair<std::string, double>>& data) {
    if (std::ifstream(filename)) {
        std::remove(filename.c_str());
    }

    std::ofstream csv_file(filename, std::ios::out); 
    if (csv_file.is_open()) {
        csv_file << "Distribution,Average Time (s)\n"; 
        for (const auto& entry : data) {
            csv_file << entry.first << "," << entry.second << "\n";
        }
        csv_file.close();
        std::cout << "Data has been written to " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file " << filename << " for writing" << std::endl;
    }
}
