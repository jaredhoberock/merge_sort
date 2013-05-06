#pragma once


namespace block
{


template<unsigned int block_size, typename Iterator1, typename Size, typename Iterator2>
__device__
void copy_n(Iterator1 first, Size n, Iterator2 result)
{
  for(Size i = threadIdx.x; i < n; i += block_size)
  {
    result[i] = first[i];
  }
}


template<unsigned int block_size, unsigned int work_per_thread, typename Iterator1, typename Size, typename Iterator2>
__device__
void copy_n_global_to_shared(Iterator1 first, Size n, Iterator2 result)
{
  typedef typename thrust::iterator_value<Iterator1>::type value_type;

  // stage copy through registers
  value_type reg[work_per_thread];

  // avoid conditional accesses when possible
  if(n >= block_size * work_per_thread)
  {
    for(unsigned int i = 0; i < work_per_thread; ++i)
    {
      unsigned int idx = block_size * i + threadIdx.x;

      reg[i] = first[idx];
    }
  }
  else
  {
    for(unsigned int i = 0; i < work_per_thread; ++i)
    {
      unsigned int idx = block_size * i + threadIdx.x;

      if(idx < n) reg[i] = first[idx];
    }
  }

  // avoid conditional accesses when possible
  if(n >= block_size * work_per_thread)
  {
    for(unsigned int i = 0; i < work_per_thread; ++i)
    {
      unsigned int idx = block_size * i + threadIdx.x;

      result[idx] = reg[i];
    }
  }
  else
  {
    for(unsigned int i = 0; i < work_per_thread; ++i)
    {
      unsigned int idx = block_size * i + threadIdx.x;

      if(idx < n) result[idx] = reg[i];
    }
  }
}


} // end block

