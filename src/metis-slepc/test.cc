#include "matCPU.hh"

#include <metis.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <slepceps.h>

#define cStar 1.0
#define eigennum 4

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
  // std::cout << "row_ptr: ";
  // for (int i = 0; i < nrows + 1; ++i) {
  //   std::cout << row_ptr[i] << " ";
  // }
  // std::cout << std::endl;
  // std::cout << "col_indices: ";
  // for (int i = 0; i < nnz; ++i) {
  //   std::cout << col_indices[i] << " " << values[i] << " ";
  // }
  // std::cout << std::endl;

  int ncon = 1;
  int objval;
  int options[METIS_NOPTIONS];
  int nparts = 150;
  std::vector<int> part(nrows);
  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
  options[METIS_OPTION_NCUTS] = 1;
  METIS_PartGraphKway(&nrows, &ncon, row_ptr.data(), col_indices.data(), NULL,
                      NULL, NULL, &nparts, NULL, NULL, NULL, &objval,
                      part.data());
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

  // int cluster_sizes[nparts];
  // for (int i = 0; i < nparts; i++) {
  //   cluster_sizes[i] = 0;
  // }
  // for (int i = 0; i < nrows; i++) {
  //   cluster_sizes[part[i]]++;
  // }
  // for (int i = 0; i < nparts; i++) {
  //   printf("Cluster %d size: %d\n", i, cluster_sizes[i]);
  // }

  std::vector<int> globalTolocal(nrows);
  std::vector<int> count(nparts, 0);
  for (int i = 0; i < nrows; ++i) {
    globalTolocal[i] = count[part[i]];
    count[part[i]]++;
  }

  // for (int i = 0; i < nparts; ++i) {
  //   std::cout << "cluster " << i << ": " << count[i] << std::endl;
  // }

  std::vector<Mat> Ai(nparts);
  std::vector<Mat> Si(nparts);

  std::vector<std::vector<int>> Ai_row_index(nparts);
  std::vector<std::vector<int>> Ai_col_index(nparts);
  std::vector<std::vector<float>> Ai_values(nparts);

  std::vector<std::vector<int>> Si_row_index(nparts);
  std::vector<std::vector<int>> Si_col_index(nparts);
  std::vector<std::vector<float>> Si_values(nparts);

  for (int i = 0; i < nrows; ++i) {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
      if (part[i] == part[col_indices[j]]) {
        if (col_indices[j] != i) {
          Ai_row_index[part[i]].push_back(globalTolocal[i]);
          Ai_col_index[part[i]].push_back(globalTolocal[i]);
          Ai_values[part[i]].push_back(values[j]);
          Ai_row_index[part[i]].push_back(globalTolocal[i]);
          Ai_col_index[part[i]].push_back(globalTolocal[col_indices[j]]);
          Ai_values[part[i]].push_back(-values[j]);

          Si_col_index[part[i]].push_back(globalTolocal[i]);
          Si_row_index[part[i]].push_back(globalTolocal[i]);
          Si_values[part[i]].push_back(values[j] / cStar / cStar / 2);
        } else {
          Ai_row_index[part[i]].push_back(globalTolocal[i]);
          Ai_col_index[part[i]].push_back(globalTolocal[i]);
          Ai_values[part[i]].push_back(values[j]);

          Si_col_index[part[i]].push_back(globalTolocal[i]);
          Si_row_index[part[i]].push_back(globalTolocal[i]);
          Si_values[part[i]].push_back(values[j] / cStar / cStar);
        }
      }
    }
  }

  for (int i = 0; i < nparts; ++i) {
    MatCreateSeqAIJFromTriple(PETSC_COMM_SELF, count[i], count[i],
                              Ai_col_index[i].data(), Ai_row_index[i].data(),
                              Ai_values[i].data(), &Ai[i], Ai_values[i].size(),
                              PETSC_FALSE);
    PetscCall(MatAssemblyBegin(Ai[i], MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Ai[i], MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(Ai[i], MAT_SYMMETRIC, PETSC_TRUE));

    PetscCall(MatConvert(Ai[i], MATSEQAIJCUSPARSE, MAT_INPLACE_MATRIX, &Ai[i]));

    MatCreateSeqAIJFromTriple(PETSC_COMM_SELF, count[i], count[i],
                              Si_col_index[i].data(), Si_row_index[i].data(),
                              Si_values[i].data(), &Si[i], Si_values[i].size(),
                              PETSC_FALSE);
    PetscCall(MatAssemblyBegin(Si[i], MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Si[i], MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(Si[i], MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatConvert(Si[i], MATSEQAIJCUSPARSE, MAT_INPLACE_MATRIX, &Si[i]));

    // PetscCall(MatView(Si[i], PETSC_VIEWER_STDOUT_WORLD));

    EPS eps;
    PetscInt nconv;
    PetscScalar eig_val, ***arr_eig_vec;
    Vec eig_vec;
    PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, count[i], &eig_vec));
    PetscCall(VecSetFromOptions(eig_vec));

    PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
    PetscCall(EPSSetOperators(eps, Ai[i], Si[i]));
    PetscCall(EPSSetProblemType(eps, EPS_GHEP));
    PetscCall(EPSSetDimensions(eps, eigennum, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(MatCreateVecs(Ai[i], &eig_vec, NULL));
    ST st;
    PetscCall(EPSGetST(eps, &st));
    PetscCall(STSetType(st, STSHIFT));
    PetscCall(EPSSetTarget(eps, -1e-12));
    PetscCall(EPSSetOptionsPrefix(eps, "epsl1_"));
    PetscCall(EPSSetFromOptions(eps));
    PetscCall(EPSSolve(eps));
    PetscCall(EPSGetConverged(eps, &nconv));

    PetscCheck(nconv >= eigennum, PETSC_COMM_SELF, PETSC_ERR_USER,
               "Not enough converged eigenvalues found!");

    PetscCall(EPSGetEigenpair(eps, PetscInt, PetscScalar *, PetscScalar *, Vec, Vec));

    PetscCall(MatDestroy(&Ai[i]));
    PetscCall(MatDestroy(&Si[i]));
  }

  PetscCall(SlepcFinalize());
}