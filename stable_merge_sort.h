#include <iostream>
#include <thrust/system/cuda/execution_policy.h>
#include <moderngpu.cuh>
#include <util/mgpucontext.h>


struct my_policy
  : thrust::cuda::execution_policy<my_policy>
{
  my_policy()
    : ctx(mgpu::CreateCudaDevice(0))
  {}

  mgpu::ContextPtr ctx;
};


namespace stable_merge_sort_detail
{


template<int VT, typename Iterator1, typename Iterator2, typename Iterator3, typename Compare>
__device__
void SerialMerge(Iterator1 first1, Iterator1 last1,
                 Iterator2 first2, Iterator2 last2,
                 Iterator3 result,
                 Compare comp)
{ 
  typename thrust::iterator_value<Iterator1>::type aKey = *first1;
  typename thrust::iterator_value<Iterator2>::type bKey = *first2;
  
  #pragma unroll
  for(int i = 0; i < VT; ++i, ++result)
  {
    bool p = (first2 >= last2) || ((first1 < last1) && !comp(bKey, aKey));
    
    *result = p ? aKey : bKey;
    
    if(p)
    {
      ++first1;
      aKey = *first1;
    }
    else
    {
      ++first2;
      bKey = *first2;
    }
  }
}


namespace block
{


template<unsigned int block_size,
         unsigned int work_per_thread,
         typename Iterator1, typename Size1,
         typename Iterator2, typename Size2,
         typename Iterator3,
         typename Compare>
__device__ void merge(Iterator1 first1, Size1 n1,
                      Iterator2 first2, Size2 n2,
                      Iterator3 result,
                      Compare comp)
{
  int diag = work_per_thread * threadIdx.x;
  int mp = mgpu::MergePath<mgpu::MgpuBoundsLower>(first1, n1, first2, n2, diag, comp);

  // compute the ranges of the sources
  int a0tid = mp;
  int a1tid = n1;
  int b0tid = diag - mp;
  int b1tid = n2;
  
  // each thread does a local sequential merge
  typedef typename thrust::iterator_value<Iterator3>::type value_type;
  value_type local_result[work_per_thread];
  SerialMerge<work_per_thread>(first1 + a0tid, first1 + a1tid,
                               first2 + b0tid, first2 + b1tid,
                               local_result, comp);

  __syncthreads();

  // store the result
  mgpu::DeviceThreadToShared<work_per_thread>(local_result, threadIdx.x, result);
}


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
void copy_n_fast(Iterator1 first, Size n, Iterator2 result)
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


template<unsigned int block_size,
         unsigned int work_per_thread,
         typename Iterator1, typename Size1,
         typename Iterator2, typename Size2,
         typename Iterator3,
	 typename Compare>
__device__
void staged_merge(Iterator1 first1, Size1 n1,
                  Iterator2 first2, Size2 n2,
                  Iterator3 result,
                  Compare comp)
{
  typedef typename thrust::iterator_value<Iterator3>::type value_type;

  __shared__ value_type s_keys[block_size * (work_per_thread + 1)];

  // stage the input through shared memory.
  // XXX replacing copy_n_fast with copy_n results in a 10% performance hit
  //copy_n<block_size>(first1, n1, s_keys); 
  //copy_n<block_size>(first2, n2, s_keys + n1);
  copy_n_fast<block_size, work_per_thread>(first1, n1, s_keys);
  copy_n_fast<block_size, work_per_thread>(first2, n2, s_keys + n1);
  __syncthreads();

  // cooperatively merge into smem
  block::merge<block_size, work_per_thread>(s_keys, n1, s_keys + n1, n2, s_keys, comp);
  
  // store result in smem to result
  block::copy_n<block_size>(s_keys, n1 + n2, result);
  __syncthreads();
}


} // end block


// Returns (a0, a1, b0, b1) into mergesort input lists between mp0 and mp1.
__host__ __device__
int4 FindMergesortInterval(int3 frame, int coop, int block, int nv, int count, int mp0, int mp1)
{
  // Locate diag from the start of the A sublist.
  int diag = nv * block - frame.x;
  int a0 = frame.x + mp0;
  int a1 = min(count, frame.x + mp1);
  int b0 = min(count, frame.y + diag - mp0);
  int b1 = min(count, frame.y + diag + nv - mp1);
  
  // The end partition of the last block for each merge operation is computed
  // and stored as the begin partition for the subsequent merge. i.e. it is
  // the same partition but in the wrong coordinate system, so its 0 when it
  // should be listSize. Correct that by checking if this is the last block
  // in this merge operation.
  if(coop - 1 == ((coop - 1) & block))
  {
    a1 = min(count, frame.x + frame.z);
    b1 = min(count, frame.y + frame.z);
  }

  return make_int4(a0, a1, b0, b1);
}


__host__ __device__
int4 ComputeMergeRange(int aCount, int block, int coop, int num_elements_per_block, const int* mp_global)
{
  // Load the merge paths computed by the partitioning kernel.
  int mp0 = mp_global[block];
  int mp1 = mp_global[block + 1];

  // coop is the number of CTAs cooperating to merge two lists into
  // one. We round block down to the first CTA's ID that is working on this
  // merge.
  int start = ~(coop - 1) & block;
  int size = num_elements_per_block * (coop >> 1);
  int3 frame = make_int3(num_elements_per_block * start, num_elements_per_block * start + size, size);

  return FindMergesortInterval(frame, coop, block, num_elements_per_block, aCount, mp0, mp1);
}


template<typename Tuning, bool MergeSort, typename KeysIt1, 
         typename KeysIt2, typename KeysIt3,
         typename Comp>
MGPU_LAUNCH_BOUNDS
void KernelMerge(KeysIt1 aKeys_global, int aCount,
                 KeysIt2 bKeys_global, int bCount,
                 const int* mp_global, int coop, KeysIt3 keys_global,
                 Comp comp)
{
  typedef MGPU_LAUNCH_PARAMS Params;
  typedef typename std::iterator_traits<KeysIt1>::value_type KeyType;
  
  const int block_size = Params::NT;
  const int work_per_thread = Params::VT;
  const int work_per_block = block_size * work_per_thread;
  
  int block = blockIdx.x;
  
  int4 range = ComputeMergeRange(aCount, block, coop, work_per_block, mp_global);
  
  block::staged_merge<block_size, work_per_thread>(aKeys_global + range.x, range.y - range.x,
                                                   bKeys_global + range.z, range.w - range.z,
                                                   keys_global + block * work_per_block,
                                                   comp);
}


template<typename RandomAccessIterator, typename Compare>
void stable_merge_sort(my_policy &exec,
                       RandomAccessIterator first,
                       RandomAccessIterator last,
                       Compare comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type T;
  typename thrust::iterator_difference<RandomAccessIterator>::type n = last - first;

  const int block_size = 256;

  const int work_per_thread = (sizeof(T) < 8) ?  11 : 7;

  typedef mgpu::LaunchBoxVT<block_size, work_per_thread> Tuning;
  int2 launch = Tuning::GetLaunchParams(*exec.ctx);
  
  const int NV = launch.x * launch.y;
  int numBlocks = MGPU_DIV_UP(n, NV);
  int numPasses = mgpu::FindLog2(numBlocks, true);
  
  MGPU_MEM(T) destDevice = exec.ctx->Malloc<T>(n);
  T* source = thrust::raw_pointer_cast(&*first);
  T* dest = destDevice->get();
  
  mgpu::KernelBlocksort<Tuning, false>
    <<<numBlocks, launch.x, 0>>>(source, (const int*)0,
    n, (1 & numPasses) ? dest : source, (int*)0, comp);
  if(1 & numPasses) std::swap(source, dest);
  
  for(int pass = 0; pass < numPasses; ++pass)
  {
    int coop = 2<< pass;
    MGPU_MEM(int) partitionsDevice =
      mgpu::MergePathPartitions<mgpu::MgpuBoundsLower>(source, n, source, 0, NV, coop, comp, *exec.ctx);
    
    KernelMerge<Tuning, true><<<numBlocks, launch.x>>>(source, n, source, 0, partitionsDevice->get(), coop, dest, comp);

    std::swap(dest, source);
  }
}


} // end stable_merge_sort_detail


template<typename RandomAccessIterator, typename Compare>
void stable_merge_sort(my_policy &exec,
                       RandomAccessIterator first,
                       RandomAccessIterator last,
                       Compare comp)
{
  stable_merge_sort_detail::stable_merge_sort(exec, first, last, comp);
}

