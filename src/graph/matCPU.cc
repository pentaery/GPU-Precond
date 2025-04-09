#include "matCPU.hh"

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