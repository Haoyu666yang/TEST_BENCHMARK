#ifndef BINARY_READ_H
#define BINARY_READ_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

template <typename T>
bool binary_read_file(const std::string& filename, std::vector<T>& data);

#endif // BINARY_READ_H
