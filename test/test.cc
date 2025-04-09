#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/vector.h>

int main() {
  thrust::device_vector<int> d_vec(10, 0); // GPU 上的向量
  for (int i = 0; i < d_vec.size(); i++) {
    d_vec[i] = i;
  }
  thrust::host_vector<int> h_vec = d_vec; // 从 GPU 拷贝到 CPU
  for (int i = 0; i < h_vec.size(); i++) {
    std::cout << h_vec[i] << " ";
  }
  std::cout << std::endl;
  return 0;
}