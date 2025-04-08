#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int readMat(int *nrows, int *nnz, std::vector<int> &row_ptr,
            std::vector<int> &col_index, std::vector<float> &values);