#include "mfem.hpp"
#include <iomanip>

void sortCSRRows(int m, int nnz, int *csrRowPtr, int *csrColInd,
                 double *csrVal) {
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

int main() {

  mfem::Device device("cuda");
  device.Print(); // 打印设备信息
  // 1. Parse command line options.
  std::string mesh_file = "../../../data/test2.msh";
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

  std::ofstream out("../../../data/A.txt");
  if (!out) {
    std::cerr << "Error: Could not open A.txt for writing!" << std::endl;
    return 1;
  }

  out << std::fixed << std::setprecision(6);

  // Get CSR components
  const int *i = A.GetI();            // row pointers
  const int *j = A.GetJ();            // column indices
  const double *a_data = A.GetData(); // values
  int nnz = A.NumNonZeroElems();      // number of non-zero elements
  int nrows = A.Height();             // number of rows


  sortCSRRows(nrows, nnz, A.GetI(), A.GetJ(), A.GetData());

  // Write CSR format: first line is "nrows nnz", then values, column indices,
  // row pointers
  out << nrows << " " << nnz << "\n";

  // Write values
  for (int k = 0; k < nnz; k++) {
    out << a_data[k];
    if (k < nnz - 1)
      out << " ";
  }
  out << "\n";

  // Write column indices
  for (int k = 0; k < nnz; k++) {
    out << j[k];
    if (k < nnz - 1)
      out << " ";
  }
  out << "\n";

  // Write row pointers
  for (int k = 0; k <= nrows; k++) {
    out << i[k];
    if (k < nrows)
      out << " ";
  }
  out << "\n";

  out.close();

  return 0;
}