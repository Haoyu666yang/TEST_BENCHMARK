#include "write_csv.h"
#include <fstream>
#include <iostream>
#include <cstdio>
#include <sys/stat.h> // For mkdir
#include <unistd.h> // For access

void write_csv(const std::string& filename, const std::vector<std::pair<std::string, double>>& data) {
    // Extract directory from filename
    size_t last_slash_idx = filename.rfind('/');
    std::string directory;
    if (std::string::npos != last_slash_idx) {
        directory = filename.substr(0, last_slash_idx);
    }

    // Check if directory exists, if not, create it
    if (!directory.empty() && access(directory.c_str(), F_OK) == -1) {
        std::cerr << "Creating directory: " << directory << std::endl;
        mkdir(directory.c_str(), 0755); // Create directory with rwx-rx-rx permissions
    }

    // If file exists, remove it
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
