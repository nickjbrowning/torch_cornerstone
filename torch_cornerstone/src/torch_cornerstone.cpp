#include <iostream>
#include <mpi.h>
#include <torch/script.h>

#define USE_CUDA

#include "cstone/domain/domain.hpp"
#include "cstone/util/reallocate.hpp"

using namespace std;
using namespace cstone;

void test_link() { std::cout << "(｡◕‿◕｡)" << std::endl; }

template <class KeyType, class T>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
domain_sync(torch::Tensor x, torch::Tensor y, torch::Tensor z, torch::Tensor h,
            int64_t numRanks, int64_t rank) {

  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat64, "Expected float64");
  TORCH_CHECK(x.sizes() == y.sizes() && y.sizes() == z.sizes() &&
                  z.sizes() == h.sizes(),
              "All tensors must have the same size");

  // Number of particles
  LocalIndex numParticles = x.size(0);

  // Simulation box
  Box<T> box(0, 1);
  unsigned bucketSize = 64, bucketSizeFocus = 8;
  float theta = 0.5;

  T *x_ptr = x.data_ptr<T>();
  T *y_ptr = y.data_ptr<T>();
  T *z_ptr = z.data_ptr<T>();
  T *h_ptr = h.data_ptr<T>();

  auto options = torch::TensorOptions().dtype(torch::kInt64).device(x.device());
  auto keys_t = torch::zeros({x.size(0)}, options);
  KeyType *keys_ptr = reinterpret_cast<KeyType *>(keys_t.data_ptr<int64_t>());

  DeviceVector<T> d_x(x_ptr, x_ptr + numParticles);
  DeviceVector<T> d_y(y_ptr, y_ptr + numParticles);
  DeviceVector<T> d_z(z_ptr, z_ptr + numParticles);
  DeviceVector<T> d_h(h_ptr, h_ptr + numParticles);
  DeviceVector<KeyType> d_keys(keys_ptr, keys_ptr + numParticles);

  Domain<KeyType, T, GpuTag> domain(rank, numRanks, bucketSize, bucketSizeFocus,
                                    theta, box);

  DeviceVector<T> s1, s2, gpuOrdering;
  domain.sync(d_keys, d_x, d_y, d_z, d_h, std::tuple{},
              std::tie(s1, s2, gpuOrdering));

  torch::Tensor new_x = torch::from_blob(
      d_x.data(), {d_x.size()},
      torch::TensorOptions().dtype(torch::kFloat64).device(x.device()));

  torch::Tensor new_y = torch::from_blob(
      d_y.data(), {d_y.size()},
      torch::TensorOptions().dtype(torch::kFloat64).device(y.device()));

  torch::Tensor new_z = torch::from_blob(
      d_z.data(), {d_z.size()},
      torch::TensorOptions().dtype(torch::kFloat64).device(z.device()));

  return {new_x, new_y, new_z};
}

TORCH_LIBRARY(torch_cornerstone, m) {
  m.def("test_link", &test_link);
  m.def("domain_sync", &domain_sync<uint64_t, double>);
}
