#include "graphLaplacian.hh"

int main() {

  SpMat<float> *A = matGen<float>();

  std::vector<int> L_row(A->size(), 0);
  std::vector<int> L_col((A->nonzeros() - A->size()) / 2, 0);
  std::vector<float> L_Value((A->nonzeros() - A->size()) / 2, 0);
  std::vector<float> M_Value(A->size(), 0);
  std::vector<int> part(A->size(), 0);

  graphPartition(A, L_row, L_col, L_Value, M_Value, part);

  return 0;
}