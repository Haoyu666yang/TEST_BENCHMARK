#ifndef WRITE_CSV_H
#define WRITE_CSV_H

#include <string>
#include <vector>

void write_csv(const std::string& filename, const std::vector<std::pair<std::string, double>>& data);

#endif // WRITE_CSV_H
