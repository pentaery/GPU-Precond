#include "matCPU.hh"

#include <metis.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <slepceps.h>

#define cStar 1.0

int main(int argc, char *argv[]) {

  PetscCall(SlepcInitialize(&argc, &argv, NULL, NULL));

  Mat A;

  int nrows, nnz;
  std::vector<float> values(1);
  std::vector<int> col_indices(1);
  std::vector<int> row_ptr(1);

  readMat(&nrows, &nnz, row_ptr, col_indices, values);

  PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, nrows, nrows,
    row_ptr.data(), col_indices.data(),
    values.data(), &A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));

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

  std::vector<std::vector<int>> localToGlobal(nparts);
  for (int i = 0; i < nparts; ++i) {
    localToGlobal[i].resize(count[i]);
  }
  for (int i = 0; i < nrows; ++i) {
    localToGlobal[part[i]][globalTolocal[i]] = i;
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

  int eigennum = 4;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-eigennum", &eigennum, NULL));


  Mat R;
  std::vector<int> nonzeros(nrows * eigennum, 0);
  for(int i = 0; i < nrows; ++i) {
    for (int j = 0; j < eigennum; ++j) {
      nonzeros[i * eigennum + j] = count[part[i]];
    }
  }
  PetscCall(MatCreateSeqAIJCUSPARSE(PETSC_COMM_SELF, nrows, nparts * eigennum, 0, nonzeros.data(), &R));

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
    PetscScalar eig_val, *arr_eig_vec;
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

    for(int j = 0; j < eigennum; ++j) {
      int *col;
      *col = i * eigennum + j;
      PetscCall(EPSGetEigenpair(eps, j, &eig_val, NULL, eig_vec, NULL));
      // std::cout << "Eigenvalue: " << eig_val << " ";
      PetscCall(VecGetArray(eig_vec, &arr_eig_vec));
      PetscCall(MatSetValues(R, count[i], localToGlobal[i].data(), 1, col, 
                            arr_eig_vec, INSERT_VALUES));
    }
    // std::cout << std::endl;





    PetscCall(MatDestroy(&Ai[i]));
    PetscCall(MatDestroy(&Si[i]));
  }

  std::cout << "Finish computing eigenvalues!" << std::endl;

  PetscCall(MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY));





  KSP ksp;
  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCall(
    KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));

  PetscCall(KSPSetFromOptions(ksp));

  Vec rhs;
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, nrows, &rhs));
  PetscCall(VecSet(rhs, 1.0));

  Vec x;
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, nrows, &x));
  PetscCall(VecSet(x, 0.0));



  PC pc;
  PetscCall(KSPGetPC(ksp, &pc));


  KSP kspCoarse, kspSmoother;
  PC pcCoarse, pcSmoother;
  // 设置二层multigrid
  PetscCall(PCSetType(pc, PCMG));
  PetscCall(PCMGSetLevels(pc, 2, NULL));
  // 设为V-cycle
  PetscCall(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
  PetscCall(PCMGSetCycleType(pc, PC_MG_CYCLE_V));
  PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH));
  // 设置coarse solver
  PetscCall(PCMGGetCoarseSolve(pc, &kspCoarse));
  PetscCall(KSPSetType(kspCoarse, KSPPREONLY));
  PetscCall(KSPGetPC(kspCoarse, &pcCoarse));
  PetscCall(PCSetType(pcCoarse, PCLU));
  PetscCall(PCFactorSetMatSolverType(pcCoarse, MATSOLVERSUPERLU_DIST));
  PetscCall(KSPSetErrorIfNotConverged(kspCoarse, PETSC_TRUE));
  // 设置一阶smoother
  PetscCall(PCMGGetSmoother(pc, 1, &kspSmoother));
  PetscCall(KSPGetPC(kspSmoother, &pcSmoother));
  PetscCall(PCSetType(pcSmoother, PCBJACOBI));
  PetscCall(KSPSetErrorIfNotConverged(kspSmoother, PETSC_TRUE));
  // 设置Prolongation
  // PetscCall(PCMGSetInterpolation(pc, 1, test2.Rc));
  PetscCall(PCShellSetName(
      pc, "3levels-MG-via-GMsFEM-with-velocity-elimination"));

  PetscCall(KSPSolve(ksp, rhs, x));

  PetscCall(SlepcFinalize());
}