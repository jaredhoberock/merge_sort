#pragma once

#include <moderngpu.cuh>
#include "copy.h"
#include "merge.h"
#include <thrust/copy.h>
#include <thrust/detail/seq.h>
#include <thrust/detail/minmax.h>
#include <thrust/detail/swap.h>


namespace static_stable_odd_even_transpose_sort_detail
{


template<int i, int n>
struct impl
{
  template<typename Iterator, typename Compare>
  static __device__
  void do_it(Iterator keys, Compare comp)
  {
    #pragma unroll
    for(int j = 1 & i; j < n - 1; j += 2)
    {
      if(comp(keys[j + 1], keys[j]))
      {
        using thrust::swap;

      	swap(keys[j], keys[j + 1]);
      }
    }

    impl<i + 1, n>::do_it(keys, comp);
  }
};


template<int i>
struct impl<i,i>
{
  template<typename Iterator, typename Compare>
  static __device__
  void do_it(Iterator, Compare) {}
};


} // end static_stable_odd_even_transpose_sort_detail


template<int n, typename RandomAccessIterator, typename Compare>
__device__
void static_stable_sort(RandomAccessIterator keys, Compare comp)
{
  static_stable_odd_even_transpose_sort_detail::impl<0,n>::do_it(keys, comp);
}

namespace block
{


template<unsigned int block_size, unsigned int work_per_thread, typename Iterator, typename Size, typename Compare>
__device__
void bounded_inplace_merge_adjacent_partitions(Iterator first,
                                               Size n,
                                               Compare comp)
{
  typedef typename thrust::iterator_value<Iterator>::type value_type;

  // the end of the input
  Iterator last = first + n;

  for(Size num_threads_per_merge = 2; num_threads_per_merge <= block_size; num_threads_per_merge *= 2)
  {
    // find the index of the first array this thread will merge
    Size list = ~(num_threads_per_merge - 1) & threadIdx.x;
    Size diag = min(n, work_per_thread * ((num_threads_per_merge - 1) & threadIdx.x));
    Size input_start = work_per_thread * list;

    // the size of each of the two input arrays we're merging
    Size input_size = work_per_thread * (num_threads_per_merge / 2);

    // find the limits of the partitions of the input this group of threads will merge
    Iterator partition_first1 = thrust::min(last, first + input_start);
    Iterator partition_first2 = thrust::min(last, partition_first1 + input_size); 
    Iterator partition_last2  = thrust::min(last, partition_first2 + input_size);

    Size mp = merge_path(diag,
                         partition_first1, partition_first2 - partition_first1,
                         partition_first2, partition_last2  - partition_first2,
                         comp);

    // each thread merges sequentially locally
    value_type local_result[work_per_thread];
    sequential_bounded_merge<work_per_thread>(partition_first1 + mp,        partition_first2,
                                              partition_first2 + diag - mp, partition_last2,
                                              local_result,
                                              comp);

    __syncthreads();

    // store local results
    thrust::copy_n(thrust::seq, local_result, work_per_thread, first + threadIdx.x * work_per_thread);

    __syncthreads();
  }
}


template<unsigned int block_size, unsigned int work_per_thread, typename RandomAccessIterator, typename Size, typename Compare>
__device__
void bounded_stable_sort(RandomAccessIterator first,
                         Size n,
                         Compare comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;

  // each thread creates a local copy of its partition of the array
  value_type local_keys[work_per_thread];
  thrust::copy_n(thrust::seq, first + threadIdx.x * work_per_thread, work_per_thread, local_keys);
  
  // if we're in the final partial tile, fill the remainder of the local_keys with with the max value
  unsigned int local_first = work_per_thread * threadIdx.x;
  if(local_first + work_per_thread > n && local_first < n)
  {
    value_type max_key = local_keys[0];

    #pragma unroll
    for(unsigned int i = 1; i < work_per_thread; ++i)
    {
      if(local_first + i < n)
      {
        max_key = comp(max_key, local_keys[i]) ? local_keys[i] : max_key;
      }
    }
    
    // fill in the uninitialized elements with max_key
    #pragma unroll
    for(unsigned int i = 0; i < work_per_thread; ++i)
    {
      if(local_first + i >= n)
      {
        local_keys[i] = max_key;
      }
    }
  }

  // stable sort the keys in the thread.
  if(work_per_thread * threadIdx.x < n)
  {
    static_stable_sort<work_per_thread>(local_keys, comp);
  }
  
  // Store the locally sorted keys into shared memory.
  thrust::copy_n(thrust::seq, local_keys, work_per_thread, first + threadIdx.x * work_per_thread);
  __syncthreads();

  block::bounded_inplace_merge_adjacent_partitions<block_size,work_per_thread>(first, n, comp);
}


} // end block


template<unsigned int block_size,
         unsigned int work_per_thread,
         bool HasValues,
         typename KeyIt1,
         typename KeyIt2, 
         typename ValIt1,
         typename ValIt2,
         typename Comp>
__global__
void KernelBlocksort(KeyIt1 keysSource_global,
                     ValIt1 valsSource_global,
                     int count,
                     KeyIt2 keysDest_global, 
                     ValIt2 valsDest_global,
                     Comp comp)
{
  typedef typename std::iterator_traits<KeyIt1>::value_type KeyType;
  typedef typename std::iterator_traits<ValIt1>::value_type ValType;
  
  const int NT = block_size;
  const int VT = work_per_thread;
  const int NV = NT * VT;

  __shared__ KeyType s_keys[NT * (VT + 1)];
  
  int block = blockIdx.x;
  int gid = NV * block;
  int tile_size = min(NV, count - gid);
  
  // load input tile into smem
  ::block::copy_n_global_to_shared<NT,VT>(keysSource_global + gid, tile_size, s_keys);
  __syncthreads();

  // sort input in smem
  ::block::bounded_stable_sort<NT,VT>(s_keys, tile_size, comp);
  
  // store result to gmem
  ::block::copy_n<block_size>(s_keys, tile_size, keysDest_global + gid);
  __syncthreads();
}


template<unsigned int block_size,
         unsigned int work_per_thread,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename Compare>
void stable_sort_each_copy(RandomAccessIterator1 first, RandomAccessIterator1 last,
                           RandomAccessIterator2 result,
                           Compare comp)
{
  typename thrust::iterator_difference<RandomAccessIterator1>::type n = last - first;
  int num_blocks = MGPU_DIV_UP(n, block_size * work_per_thread);
  KernelBlocksort<block_size, work_per_thread, false><<<num_blocks, block_size>>>(first, (int*)0, n, result, (int*)0, comp);
}

