#include <iostream>
#include <torch/script.h>

#include "cstone/domain/domain.hpp"
#include "cstone/util/reallocate.hpp"

using namespace std;

void test_link() { std::cout << "(｡◕‿◕｡)" << std::endl; }

TORCH_LIBRARY(torch_cornerstone, m) {
  m.def("test_link", &test_link);
}
