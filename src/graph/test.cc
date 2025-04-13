#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include "matCPU.hh"

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {

  int nrows, nnz;
  std::vector<float> values(1);    
  std::vector<int> col_indices(1); 
  std::vector<int> row_ptr(1);     

  readMat(&nrows, &nnz, row_ptr, col_indices, values);
  matDecompose2LM(&nrows, &nnz, row_ptr, col_indices, values);

  std::cout << "nrows: " << nrows << std::endl;
  std::cout << "nnz: " << nnz << std::endl;

  std::vector<float> weights(nnz, 1.0f); // 权重数组（初始化为 1.0）
  int num_vertices = nrows;
  int num_edges = nnz;

  thrust::device_vector<float> d_weights(weights);
  thrust::device_vector<int> d_row_ptr(row_ptr);
  thrust::device_vector<int> d_col_indices(col_indices);
  thrust::host_vector<int> h_clustering(num_vertices, 0.f);
  thrust::device_vector<int> d_clustering(h_clustering);

  cugraph::legacy::GraphCSRView<int, int, float> graph;
  graph.offsets = thrust::raw_pointer_cast(d_row_ptr.data());
  graph.indices = thrust::raw_pointer_cast(d_col_indices.data());
  graph.edge_data = thrust::raw_pointer_cast(d_weights.data());
  graph.number_of_vertices = num_vertices;
  graph.number_of_edges = num_edges;

  // 设置聚类参数
  int n_clusters = 120;
  int n_eig_vects = n_clusters;
  float evs_tolerance = 0.0001;
  int evs_max_iter = 1000;
  float kmean_tolerance = 0.0001;
  int kmean_max_iter = 1000;

  // 调用谱聚类函数
  cugraph::ext_raft::spectralModularityMaximization(
      graph, n_clusters, n_eig_vects, evs_tolerance, evs_max_iter,
      kmean_tolerance, kmean_max_iter,
      thrust::raw_pointer_cast(d_clustering.data()));

  int clustering[num_vertices];
  checkCudaError(cudaMemcpy(clustering,
                            thrust::raw_pointer_cast(d_clustering.data()),
                            num_vertices * sizeof(int), cudaMemcpyDeviceToHost),
                 "Failed to copy clustering");

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

  return 0;
}