#include "stable_merge_sort.h"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <vector>
#include <cassert>


struct hash_functor
{
  __host__ __device__
  unsigned int operator()(unsigned int x)
  {
    x = (x+0x7ed55d16) + (x<<12);
    x = (x^0xc761c23c) ^ (x>>19);
    x = (x+0x165667b1) + (x<<5);
    x = (x+0xd3a2646c) ^ (x<<9);
    x = (x+0xfd7046c5) + (x<<3);
    x = (x^0xb55a4f09) ^ (x>>16);
    return x;
  }
};


template<typename Vector>
void generate_random_data(Vector &vec)
{
  thrust::tabulate(vec.begin(), vec.end(), hash_functor());
}


void do_it(my_policy &exec, size_t n)
{
  std::vector<int> h_data(n);
  generate_random_data(h_data);

  thrust::device_vector<int> d_data = h_data;

  std::stable_sort(h_data.begin(), h_data.end());

  ::stable_merge_sort(exec, d_data.begin(), d_data.end(), thrust::less<int>());

  cudaError_t error = cudaGetLastError();

  if(error)
  {
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
  }

  assert(h_data == d_data);
}


int main()
{
  my_policy exec;

  for(size_t n = 1; n <= 1 << 20; n <<= 1)
  {
    std::cout << "Testing n = " << n << std::endl;
    do_it(exec, n);
  }

  return 0;
}

