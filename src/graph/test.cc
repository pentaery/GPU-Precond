#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {

  std::ifstream in("../../../data/A.txt");
  if (!in) {
    std::cerr << "Error: Could not open A.txt for reading!" << std::endl;
    return 1;
  }

  // 2. 读取第一行：行数和非零元素个数
  int nrows, nnz;
  in >> nrows >> nnz;
  in.ignore(); // 跳过换行符

  // 3. 分配存储空间
  std::vector<float> values(nnz);      // 值数组（使用 float 类型）
  std::vector<int> col_indices(nnz);   // 列索引数组
  std::vector<int> row_ptr(nrows + 1); // 行指针数组（nrows + 1 个元素）

  // 4. 读取值数组
  std::string line;
  std::getline(in, line); // 读取整行
  std::istringstream iss_values(line);
  for (int i = 0; i < nnz; i++) {
    if (!(iss_values >> values[i])) {
      std::cerr << "Error reading values at index " << i << std::endl;
      return 1;
    }
  }

  // 5. 读取列索引数组
  std::getline(in, line);
  std::istringstream iss_cols(line);
  for (int i = 0; i < nnz; i++) {
    if (!(iss_cols >> col_indices[i])) {
      std::cerr << "Error reading column indices at index " << i << std::endl;
      return 1;
    }
  }

  // 6. 读取行指针数组
  std::getline(in, line);
  std::istringstream iss_rows(line);
  for (int i = 0; i <= nrows; i++) {
    if (!(iss_rows >> row_ptr[i])) {
      std::cerr << "Error reading row pointers at index " << i << std::endl;
      return 1;
    }
  }

  // 7. 关闭文件
  in.close();

  int offsets[] = {};
  int indices[] = {0, 1, 2, 3}; // 修正为 4 条边
  float weights[163476] = {1};
  for (int i = 0; i < 163476; i++) {
    weights[i] = 1;
  }
  int num_vertices = 23526;
  int num_edges = 163476; // 修正边数为 4

  // 在设备上分配内存
  int *d_offsets, *d_indices;
  float *d_weights;
  int *d_clustering;

  checkCudaError(cudaMalloc(&d_offsets, (num_vertices + 1) * sizeof(int)),
                 "Failed to allocate d_offsets");
  checkCudaError(cudaMalloc(&d_indices, num_edges * sizeof(int)),
                 "Failed to allocate d_indices");
  checkCudaError(cudaMalloc(&d_weights, num_edges * sizeof(float)),
                 "Failed to allocate d_weights");
  checkCudaError(cudaMalloc(&d_clustering, num_vertices * sizeof(int)),
                 "Failed to allocate d_clustering");

  // 将数据从主机复制到设备
  checkCudaError(cudaMemcpy(d_offsets, row_ptr.data(),
                            (num_vertices + 1) * sizeof(int),
                            cudaMemcpyHostToDevice),
                 "Failed to copy offsets");
  checkCudaError(cudaMemcpy(d_indices, col_indices.data(),
                            num_edges * sizeof(int), cudaMemcpyHostToDevice),
                 "Failed to copy indices");
  checkCudaError(cudaMemcpy(d_weights, weights, num_edges * sizeof(float),
                            cudaMemcpyHostToDevice),
                 "Failed to copy weights");

  // 设置图
  cugraph::legacy::GraphCSRView<int, int, float> graph;
  graph.offsets = d_offsets;
  graph.indices = d_indices;
  graph.edge_data = d_weights;
  graph.number_of_vertices = num_vertices;
  graph.number_of_edges = num_edges;

  // 设置聚类参数
  int n_clusters = 150;
  int n_eig_vects = 150;
  float evs_tolerance = 0.00001;
  int evs_max_iter = 1000;
  float kmean_tolerance = 0.00001;
  int kmean_max_iter = 1000;

  // 调用谱聚类函数
  cugraph::ext_raft::spectralModularityMaximization(
      graph, n_clusters, n_eig_vects, evs_tolerance, evs_max_iter,
      kmean_tolerance, kmean_max_iter, d_clustering);

  // cugraph::ext_raft::analyzeClustering_modularity(graph, n_clusters, NULL,
  // d_clustering);

  // 将结果复制回主机
  int clustering[num_vertices];
  checkCudaError(cudaMemcpy(clustering, d_clustering,
                            num_vertices * sizeof(int), cudaMemcpyDeviceToHost),
                 "Failed to copy clustering");

  // 打印聚类结果到clustering.txt 每一行代表一个顶点的聚类标签
  std::ofstream out("clustering.txt");
  if (!out) {
    std::cerr << "Error: Could not open clustering.txt for writing!"
              << std::endl;
    return 1;
  }
  for (int i = 0; i < num_vertices; i++) {
    out << clustering[i] << std::endl;
  }
  out.close();

  // 统计每个簇的大小
  int cluster_sizes[n_clusters];
  for (int i = 0; i < n_clusters; i++) {
    cluster_sizes[i] = 0;
  }
  for (int i = 0; i < num_vertices; i++) {
    cluster_sizes[clustering[i]]++;
  }
  for (int i = 0; i < n_clusters; i++) {
    printf("Cluster %d size: %d\n", i, cluster_sizes[i]);
  }

  // 释放设备内存
  cudaFree(d_offsets);
  cudaFree(d_indices);
  cudaFree(d_weights);
  cudaFree(d_clustering);

  return 0;
}