#include "matCPU.hh"

#include <metis.h>



int main(int argc, char *argv[]) {

  PetscCall(SlepcInitialize(&argc, &argv, NULL, NULL));

  int nrows, nnz;
  std::vector<float> values(1);
  std::vector<int> col_indices(1);
  std::vector<int> row_ptr(1);

  readMat(&nrows, &nnz, row_ptr, col_indices, values);
  matDecompose2LM(&nrows, &nnz, row_ptr, col_indices, values);

  std::cout << "nrows: " << nrows << std::endl;
  std::cout << "nnz: " << nnz << std::endl;

  int ncon = 1;
  int objval;
  int options[METIS_NOPTIONS];
  int nparts = 150;
  std::vector<int> part(nrows);
  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
  options[METIS_OPTION_NCUTS] = 1;
  METIS_PartGraphKway(&nrows, &ncon, row_ptr.data(), col_indices.data(), NULL, NULL, NULL,
                      &nparts, NULL, NULL, NULL, &objval, part.data());
  std::cout << "Objective for the partition is " << objval << std::endl;

  std::ofstream out("clustering.txt");
  if (!out) {
    std::cerr << "Error: Could not open clustering.txt for writing!"
              << std::endl;
    return 1;
  }
  for (int i = 0; i < nrows; i++) {
    out << part[i] << std::endl;
  }
  out.close();


  int cluster_sizes[nparts];
  for (int i = 0; i < nparts; i++) {
    cluster_sizes[i] = 0;
  }
  for (int i = 0; i < nrows; i++) {
    cluster_sizes[part[i]]++;
  }
  for (int i = 0; i < nparts; i++) {
    printf("Cluster %d size: %d\n", i, cluster_sizes[i]);
  }

  
  PetscCall(SlepcFinalize());
}