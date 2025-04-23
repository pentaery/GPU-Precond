#include "matCPU.hh"
#include <petscerror.h>
#include <petscmat.h>

// void processDataThrust(thrust::device_vector<int> &globalTolocal,
//                        thrust::device_vector<int> &count,
//                        thrust::device_vector<int> &part, int nvtxs,
//                        int nparts) {
//   // 1. 初始化 count 为 0
//   thrust::fill(count.begin(), count.end(), 0);

//   // 2. 创建临时数组存储每个顶点的计数（都为1）
//   thrust::device_vector<int> ones(nvtxs);
//   thrust::fill(ones.begin(), ones.end(), 1);

//   // 3. 按 part 进行 reduce_by_key，计算每个分区的计数
//   thrust::device_vector<int> temp_count(nparts);
//   thrust::reduce_by_key(part.begin(), part.end(), ones.begin(),
//                         thrust::discard_iterator<>(), temp_count.begin());

//   // 4. 将 temp_count 拷贝到 count
//   thrust::copy(temp_count.begin(), temp_count.end(), count.begin());

//   // 5. 对每个分区内的元素进行前缀和，得到 local index
//   thrust::device_vector<int> temp_indices(nvtxs);
//   thrust::exclusive_scan_by_key(part.begin(), part.end(), ones.begin(),
//                                 temp_indices.begin());

//   // 6. 将结果写入 globalTolocal
//   thrust::copy(temp_indices.begin(), temp_indices.end(),
//   globalTolocal.begin());
// }

int readMat(int *nrows, int *nnz, std::vector<int> &row_ptr,
            std::vector<int> &col_index, std::vector<float> &values) {
  std::ifstream in("../../../data/A.txt");
  if (!in) {
    std::cerr << "Error: Could not open A.txt for reading!" << std::endl;
    return 1;
  }

  in >> *nrows >> *nnz;
  in.ignore(); // 跳过换行符

  row_ptr.resize(*nrows + 1);
  col_index.resize(*nnz);
  values.resize(*nnz);

  // 4. 读取值数组
  std::string line;
  std::getline(in, line); // 读取整行
  std::istringstream iss_values(line);
  for (int i = 0; i < *nnz; i++) {
    if (!(iss_values >> values[i])) {
      std::cerr << "Error reading values at index " << i << std::endl;
      return 1;
    }
  }

  // 5. 读取列索引数组
  std::getline(in, line);
  std::istringstream iss_cols(line);
  for (int i = 0; i < *nnz; i++) {
    if (!(iss_cols >> col_index[i])) {
      std::cerr << "Error reading column indices at index " << i << std::endl;
      return 1;
    }
  }

  // 6. 读取行指针数组
  std::getline(in, line);
  std::istringstream iss_rows(line);
  for (int i = 0; i <= *nrows; i++) {
    if (!(iss_rows >> row_ptr[i])) {
      std::cerr << "Error reading row pointers at index " << i << std::endl;
      return 1;
    }
  }

  // 7. 关闭文件
  in.close();

  return 0;
}

void matDecompose2LM(int *nrows, int *nnz, std::vector<int> &row_ptr,
                     std::vector<int> &col_index, std::vector<float> &values) {
  // Step 1: Calculate row sums and update diagonal elements
  std::vector<float> row_sums(*nrows, 0.0f);

  // Calculate row sums
  for (int i = 0; i < *nrows; i++) {
    int row_start = row_ptr[i];
    int row_end = row_ptr[i + 1];

    for (int j = row_start; j < row_end; j++) {
      row_sums[i] += values[j];
    }
  }

  // Update diagonal elements
  for (int i = 0; i < *nrows; i++) {
    int row_start = row_ptr[i];
    int row_end = row_ptr[i + 1];

    for (int j = row_start; j < row_end; j++) {
      if (col_index[j] == i) { // Diagonal element
        values[j] = row_sums[i];
        break;
      }
    }
  }

  // Step 2: In-place removal of lower triangular elements and scaling

  // int new_nnz = 0;
  // int current_pos = 0;

  // for (int i = 0; i < *nrows; i++) {
  //   int row_start = row_ptr[i];
  //   int row_end = row_ptr[i + 1];
  //   int new_row_start = current_pos;

  //   for (int j = row_start; j < row_end; j++) {
  //     if (col_index[j] >= i) { // Keep upper triangular elements
  //       // Move elements to their new positions
  //       if (current_pos != j) {
  //         col_index[current_pos] = col_index[j];
  //         values[current_pos] = values[j] * -2.0f;
  //       } else {
  //         values[current_pos] *= -2.0f;
  //       }
  //       current_pos++;
  //     }
  //   }

  //   row_ptr[i] = new_row_start;
  //   new_nnz = current_pos;
  // }

  // // Update the last row_ptr entry
  // row_ptr[*nrows] = new_nnz;

  // // Resize the vectors to remove unused space
  // col_index.resize(new_nnz);
  // values.resize(new_nnz);
  // *nnz = new_nnz;
}

// PetscErrorCode formAUX(std::vector<int> &row_ptr, std::vector<int> &col_index,
//                        std::vector<float> &values, int nrows, int nnz) {
//   PetscFunctionBegin;

//   // 调用 readMat 函数读取矩阵数据
//   if (readMat(&nrows, &nnz, row_ptr, col_index, values) != 0) {
//     std::cerr << "Error reading matrix data!" << std::endl;
//     PetscFunctionReturn(PETSC_ERR_FILE_OPEN);
//   }

//   // 创建 PETSc 矩阵
//   Mat A;
//   PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, nrows, nrows, row_ptr.data(), col_index.data(), values.data(), &A));
//   // 完成矩阵装配
//   PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
//   PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

//   // 转换为 CUDA 支持的矩阵类型
//   PetscCall(MatConvert(A, MATSEQAIJCUSPARSE, MAT_INPLACE_MATRIX, &A));

//   // 打印矩阵信息（可选）
//   PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

//   // 销毁矩阵
//   PetscCall(MatDestroy(&A));

//   PetscFunctionReturn(0);
// }