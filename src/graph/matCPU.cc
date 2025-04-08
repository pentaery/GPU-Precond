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