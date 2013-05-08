#include "stable_merge_sort_by_key.h"
#include "time_invocation_cuda.hpp"
#include <moderngpu.cuh>
#include <util/mgpucontext.h>
#include <thrust/device_vector.h>
#include <thrust/tabulate.h>
#include <algorithm>
#include <vector>
#include <cassert>


typedef int T;


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


void do_it(cached_allocator &alloc, size_t n)
{
  thrust::host_vector<T> h_keys(n), h_values(n);
  generate_random_data(h_keys);
  generate_random_data(h_values);

  thrust::device_vector<T> d_keys = h_keys;
  thrust::device_vector<T> d_values = h_values;

  thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin());

  ::stable_merge_sort_by_key(thrust::cuda::par(alloc), d_keys.begin(), d_keys.end(), d_values.begin(), thrust::less<T>());

  cudaError_t error = cudaGetLastError();

  if(error)
  {
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
  }

  assert(h_keys == d_keys);
  assert(h_values == d_values);
}


void my_sort(cached_allocator *alloc, thrust::device_vector<T> *keys, thrust::device_vector<T> *values)
{
  generate_random_data(*keys);
  generate_random_data(*values);

  stable_merge_sort_by_key(thrust::cuda::par(*alloc), keys->begin(), keys->end(), values->begin(), thrust::less<T>());
}


void sean_sort(mgpu::ContextPtr *ctx, thrust::device_vector<T> *keys, thrust::device_vector<T> *values)
{
  generate_random_data(*keys);
  generate_random_data(*values);

  mgpu::MergesortPairs(thrust::raw_pointer_cast(keys->data()),
                       thrust::raw_pointer_cast(values->data()),
                       keys->size(),
                       thrust::less<T>(),
                       **ctx);
}


int main()
{
  mgpu::ContextPtr ctx = mgpu::CreateCudaDevice(0);

  cached_allocator alloc;

  for(size_t n = 1; n <= 1 << 20; n <<= 1)
  {
    std::cout << "Testing n = " << n << std::endl;
    do_it(alloc, n);
  }

  for(int i = 0; i < 20; ++i)
  {
    size_t n = hash_functor()(i) % (1 << 20);

    std::cout << "Testing n = " << n << std::endl;
    do_it(alloc, n);
  }

  thrust::device_vector<T> keys(1 << 24);
  thrust::device_vector<T> values(1 << 24);

  sean_sort(&ctx, &keys, &values);
  double sean_msecs = time_invocation_cuda(20, sean_sort, &ctx, &keys, &values);

  my_sort(&alloc, &keys, &values);
  double my_msecs = time_invocation_cuda(20, my_sort, &alloc, &keys, &values);

  std::cout << "Sean's time: " << sean_msecs << " ms" << std::endl;
  std::cout << "My time: " << my_msecs << " ms" << std::endl;

  std::cout << "My relative performance: " << sean_msecs / my_msecs << std::endl;

  return 0;
}

