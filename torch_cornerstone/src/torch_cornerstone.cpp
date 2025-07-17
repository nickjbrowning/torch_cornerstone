#include <iostream>
#include <torch/script.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

void test() {}

TORCH_LIBRARY(torch_cornerstone, m) { m.def("test", &test); }
