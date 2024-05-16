#ifndef WRITE_CSV_H
#define WRITE_CSV_H

#include <vector>
#include <string>

void write_csv(const std::string& filename, const std::vector<std::pair<std::string, double>>& data);

#endif
