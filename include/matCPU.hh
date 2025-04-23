#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <slepc.h>
#include <slepcsys.h>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

// #include <amgx_c.h>

int readMat(int *nrows, int *nnz, std::vector<int> &row_ptr,
            std::vector<int> &col_index, std::vector<float> &values);

void matDecompose2LM(int *nrows, int *nnz, std::vector<int> &row_ptr,
                     std::vector<int> &col_index, std::vector<float> &values);

void processDataThrust(thrust::device_vector<int> &globalTolocal,
                       thrust::device_vector<int> &count,
                       thrust::device_vector<int> &part, int nvtxs, int nparts);

PetscErrorCode formAUX(std::vector<int> &row_ptr, std::vector<int> &col_index,
             std::vector<float> &values, int nrows, int nnz);