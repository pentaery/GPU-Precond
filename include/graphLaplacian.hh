#include "mfem.hpp"
#include <cstddef>
#include <fstream>
#include <iostream>
#include <linalg/sparsemat.hpp>
#include <metis.h>

template <typename T> class SpMat {
private:
  int n;
  int nnz;
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<T> values;

public:
  SpMat(int size) : n(size), nnz(0) { row_ptr.resize(n + 1, 0); }

  SpMat(int size, int nnz, int *I, int *J, double *V)
      : n(size), nnz(nnz), row_ptr(I, I + size), col_idx(J, J + nnz) {
    values.resize(nnz);
    for (int i = 0; i < nnz; ++i) {
      values[i] = static_cast<T>(V[i]);
    }
  }

  int size() const { return n; }

  int nonzeros() const { return nnz; }

  std::vector<int> &getRowPtr() { return row_ptr; }
  std::vector<int> &getColIdx() { return col_idx; }
  std::vector<T> &getValues() { return values; }

  void print() const {
    std::cout << "CSR Matrix (" << n << " x " << n << "), NNZ = " << nnz
              << "\n";
    std::cout << "Row Ptr: ";
    for (int x : row_ptr)
      std::cout << x << " ";
    std::cout << "\nCol Idx: ";
    for (int x : col_idx)
      std::cout << x << " ";
    std::cout << "\nValues: ";
    for (double x : values)
      std::cout << x << " ";
    std::cout << "\n";
  }
};

template <typename T> SpMat<T> *matGen() {
  mfem::Device device("cuda");
  device.Print(); // 打印设备信息
  // 1. Parse command line options.
  std::string mesh_file = "../../rect";
  int order = 1;

  mfem::OptionsParser args(0, 0);
  // args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  // args.AddOption(&order, "-o", "--order", "Finite element polynomial
  //   degree");
  // args.ParseCheck();

  // 2. Read the mesh from the given mesh file, and refine once uniformly.
  mfem::Mesh mesh(mesh_file);
  // mesh.UniformRefinement();
  std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
  std::cout << "Dimension: " << mesh.Dimension() << std::endl;
  // 3. Define a finite element space on the mesh. Here we use H1 continuous
  //    high-order Lagrange finite elements of the given order.
  mfem::H1_FECollection fec(order, mesh.Dimension());
  mfem::FiniteElementSpace fespace(&mesh, &fec);
  std::cout << "Number of unknowns: " << fespace.GetTrueVSize() << std::endl;

  // 4. Extract the list of all the boundary DOFs. These will be marked as
  //    Dirichlet in order to enforce zero boundary conditions.
  mfem::Array<int> boundary_dofs;
  fespace.GetBoundaryTrueDofs(boundary_dofs);

  // 5. Define the solution x as a finite element grid function in fespace.
  //   Set
  //    the initial guess to zero, which also sets the boundary conditions.
  mfem::GridFunction x(&fespace);
  x = 0.0;

  // 6. Set up the linear form b(.) corresponding to the right-hand side.
  mfem::ConstantCoefficient one(1.0);
  mfem::LinearForm b(&fespace);
  b.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
  b.Assemble();

  // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
  mfem::BilinearForm a(&fespace);
  a.AddDomainIntegrator(new mfem::DiffusionIntegrator);
  a.Assemble();

  // 8. Form the linear system A X = B. This includes eliminating boundary
  //    conditions, applying AMR constraints, and other transformations.
  mfem::SparseMatrix A;
  mfem::Vector B, X;
  a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

  return new SpMat<float>(A.Height(), A.NumNonZeroElems(), A.GetI(), A.GetJ(),
                          A.GetData());
}

template <typename T>
void graphPartition(SpMat<T> *A, std::vector<int> &L_row,
                    std::vector<int> &L_col, std::vector<T> &L_Value,
                    std::vector<T> &M_Value, std::vector<int> &part) {
  int n = A->size();
  int nnz = A->nonzeros();
  int *I = A->getRowPtr().data();
  int *J = A->getColIdx().data();
  T *V = A->getValues().data();
  // A->print();

  int L_nnz = 0;
  L_row[0] = 0;

  for (int i = 0; i < n - 1; ++i) {
    M_Value[i] = 0;
    for (int j = I[i]; j < I[i + 1]; ++j) {
      M_Value[i] += V[j];
      if (J[j] > i && V[j] < -1e-5) {
        L_row[i + 1]++;
        L_col[L_nnz] = J[j];
        L_Value[L_nnz++] = -2 * V[j];
      }
    }
    L_row[i + 1] += L_row[i];
  }

  // L_col.resize(L_nnz);
  // L_Value.resize(L_nnz);

  // for (int i = 0; i < n; ++i) {
  //   std::cout << L_row[i] << " ";
  // }
  // std::cout << "\n";
  // for (int i = 0; i < L_nnz; ++i) {
  //   std::cout << L_col[i] << " ";
  // }
  // std::cout << "\n";
  // for (int i = 0; i < L_nnz; ++i) {
  //   std::cout << L_Value[i] << " ";
  // }
  // std::cout << "\n";

  idx_t ncon = 1;
  idx_t objval;
  idx_t options[METIS_NOPTIONS];
  int nparts = 10;
  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
  options[METIS_OPTION_NCUTS] = 1;
  METIS_PartGraphKway(&n, &ncon, L_row.data(), L_col.data(), NULL, NULL, NULL,
                      &nparts, NULL, NULL, NULL, &objval, part.data());
  std::cout << "Objective for the partition is " << objval << std::endl;


}
