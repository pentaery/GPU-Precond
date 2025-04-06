#include <algorithm>
#include <iostream>
#include <vector>

void sortCSRRows(int m, int nnz, int *csrRowPtr, int *csrColInd,
                 float *csrVal) {
  // 遍历每一行
  for (int row = 0; row < m; ++row) {
    // 获取当前行的起始和结束位置
    int start = csrRowPtr[row];
    int end = csrRowPtr[row + 1];
    int row_nnz = end - start; // 当前行的非零元素个数

    if (row_nnz <= 1) {
      // 如果行内元素少于 2 个，无需排序
      continue;
    }

    // 将列索引和值绑定为 pair 进行排序
    std::vector<std::pair<int, float>> row_pairs(row_nnz);
    for (int i = start; i < end; ++i) {
      row_pairs[i - start] = std::make_pair(csrColInd[i], csrVal[i]);
    }

    // 按列索引升序排序
    std::sort(row_pairs.begin(), row_pairs.end(),
              [](const std::pair<int, float> &a,
                 const std::pair<int, float> &b) { return a.first < b.first; });

    // 将排序后的结果写回原始数组
    for (int i = start; i < end; ++i) {
      csrColInd[i] = row_pairs[i - start].first;
      csrVal[i] = row_pairs[i - start].second;
    }
  }
}

// 测试代码
int main() {
  // 示例 CSR 矩阵：
  // 行 0: (2, 1.0), (0, 2.0)  -> 需要排序为 (0, 2.0), (2, 1.0)
  // 行 1: (1, 3.0), (0, 4.0)  -> 需要排序为 (0, 4.0), (1, 3.0)
  // 行 2: (2, 5.0)            -> 无需排序
  int m = 3;   // 行数
  int nnz = 5; // 非零元素个数
  int csrRowPtr[] = {0, 2, 4, 5};
  int csrColInd[] = {2, 0, 1, 0, 2};          // 未排序的列索引
  float csrVal[] = {1.0, 2.0, 3.0, 4.0, 5.0}; // 对应的值

  std::cout << "Before sorting:" << std::endl;
  for (int i = 0; i < m; ++i) {
    std::cout << "Row " << i << ": ";
    for (int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; ++j) {
      std::cout << "(" << csrColInd[j] << ", " << csrVal[j] << ") ";
    }
    std::cout << std::endl;
  }

  // 调用排序函数
  sortCSRRows(m, nnz, csrRowPtr, csrColInd, csrVal);

  std::cout << "After sorting:" << std::endl;
  for (int i = 0; i < m; ++i) {
    std::cout << "Row " << i << ": ";
    for (int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; ++j) {
      std::cout << "(" << csrColInd[j] << ", " << csrVal[j] << ") ";
    }
    std::cout << std::endl;
  }

  return 0;
}