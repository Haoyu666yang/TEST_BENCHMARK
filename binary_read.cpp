#include "binary_read.h"

template <typename T>
bool binary_read_file(const std::string& filename, std::vector<T>& data) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "error opening file: " << filename << std::endl;
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    data.resize(size / sizeof(T));
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        std::cerr << "Error reading file: " << filename << std::endl;
        return false;
    }

    return true;
}

// Explicit template instantiation
template bool binary_read_file<int32_t>(const std::string& filename, std::vector<int32_t>& data);
template bool binary_read_file<int64_t>(const std::string& filename, std::vector<int64_t>& data);
