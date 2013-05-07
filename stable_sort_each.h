#pragma once

#include <moderngpu.cuh>
#include "copy.h"
#include "merge.h"
#include <thrust/copy.h>
#include <thrust/detail/seq.h>


template<int NT, int VT, typename T, typename Comp>
__device__
void CTABlocksortPass(T* keys_shared, int tid, int count, int coop, T* keys, Comp comp)
{
  int list = ~(coop - 1) & tid;
  int diag = min(count, VT * ((coop - 1) & tid));
  int start = VT * list;
  int a0 = min(count, start);
  int b0 = min(count, start + VT * (coop / 2));
  int b1 = min(count, start + VT * coop);
  
  int p = mgpu::MergePath<mgpu::MgpuBoundsLower>(keys_shared + a0, b0 - a0, keys_shared + b0, b1 - b0, diag, comp);
  
  sequential_bounded_merge<VT>(keys_shared + a0 + p,        keys_shared + b0,
                               keys_shared + b0 + diag - p, keys_shared + b1,
                               keys,
                               comp);
  __syncthreads();
}


template<int NT, int VT, typename KeyType, typename Comp>
__device__
void CTABlocksortLoop(KeyType* keys_shared,
                      int tid,
                      int count, 
                      Comp comp)
{
  #pragma unroll
  for(int coop = 2; coop <= NT; coop *= 2)
  {
    KeyType keys[VT];
    ::CTABlocksortPass<NT, VT>(keys_shared, tid, count, coop, keys, comp);
    
    // Store results in shared memory in sorted order.
    thrust::copy_n(thrust::seq, keys, VT, keys_shared + tid * VT);
    __syncthreads();
  }
}


template<int NT, int VT, bool HasValues, typename KeyType, typename ValType, typename Comp>
__device__
void CTAMergesort(KeyType threadKeys[VT],
                  ValType threadValues[VT],
                  KeyType* keys_shared,
                  ValType* values_shared,
                  int count,
                  int tid, 
                  Comp comp)
{
  // Stable sort the keys in the thread.
  if(VT * tid < count)
  {
    mgpu::OddEvenTransposeSort<VT>(threadKeys, threadValues, comp);
  }
  
  // Store the locally sorted keys into shared memory.
  thrust::copy_n(thrust::seq, threadKeys, VT, keys_shared + tid * VT);
  __syncthreads();
  
  // Recursively merge lists until the entire CTA is sorted.
  ::CTABlocksortLoop<NT, VT>(keys_shared, tid, count, comp);
}


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
  union Shared {
  	KeyType keys[NT * (VT + 1)];
  	ValType values[NV];
  };
  __shared__ Shared shared;
  
  int tid = threadIdx.x;
  int block = blockIdx.x;
  int gid = NV * block;
  int count2 = min(NV, count - gid);
  
  // Load the values into thread order.
  ValType threadValues[VT];
  if(HasValues) {
    mgpu::DeviceGlobalToShared<NT, VT>(count2, valsSource_global + gid, tid, shared.values);
    mgpu::DeviceSharedToThread<VT>(shared.values, tid, threadValues);
  }
  
  // Load keys into shared memory and transpose into register in thread order.
  KeyType threadKeys[VT];
  ::block::copy_n_global_to_shared<NT,VT>(keysSource_global + gid, count2, shared.keys);
  __syncthreads();
  thrust::copy_n(thrust::seq, shared.keys + tid * VT, VT, threadKeys);
  
  // If we're in the last tile, set the uninitialized keys for the thread with
  // a partial number of keys.
  int first = VT * tid;
  if(first + VT > count2 && first < count2)
  {
    KeyType maxKey = threadKeys[0];
    #pragma unroll
    for(int i = 1; i < VT; ++i)
    	if(first + i < count2)
    		maxKey = comp(maxKey, threadKeys[i]) ? threadKeys[i] : maxKey;
    
    // Fill in the uninitialized elements with max key.
    #pragma unroll
    for(int i = 0; i < VT; ++i)
    	if(first + i >= count2) threadKeys[i] = maxKey;
  }
  
  ::CTAMergesort<NT, VT, HasValues>(threadKeys, threadValues, shared.keys, shared.values, count2, tid, comp);
  
  // Store the sorted keys to global.
  ::block::copy_n<block_size>(shared.keys, count2, keysDest_global + gid);
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

