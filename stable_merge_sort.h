#include <iostream>
#include <moderngpu.cuh>
#include <util/mgpucontext.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/memory.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/detail/detail/uninitialized.h>
#include <thrust/merge.h>
#include <thrust/tuple.h>
#include <thrust/tabulate.h>
#include <thrust/detail/minmax.h>


struct my_policy
{
  my_policy()
    : ctx(mgpu::CreateCudaDevice(0))
  {}

  mgpu::ContextPtr ctx;
};


namespace stable_merge_sort_detail2
{


// sequential merge for when we have a static bound on the size of the result
template<unsigned int result_size_bound, typename Iterator1, typename Iterator2, typename Iterator3, typename Compare>
__device__
void sequential_bounded_merge(Iterator1 first1, Iterator1 last1,
                              Iterator2 first2, Iterator2 last2,
                              Iterator3 result,
                              Compare comp)
{ 
  typename thrust::iterator_value<Iterator1>::type aKey = *first1;
  typename thrust::iterator_value<Iterator2>::type bKey = *first2;
  
  #pragma unroll
  for(int i = 0; i < result_size_bound; ++i, ++result)
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


template<typename Iterator1,
         typename Size,
         typename Iterator2,
         typename Iterator3,
         typename Compare>
__device__ void merge(unsigned int work_per_thread,
                      Iterator1 first1, Size n1,
                      Iterator2 first2, Size n2,
                      Iterator3 result,
                      Compare comp)
{
  Size diag = min(n1 + n2, work_per_thread * threadIdx.x);
  Size mp = mgpu::MergePath<mgpu::MgpuBoundsLower>(first1, n1, first2, n2, diag, comp);

  // compute the ranges of the sources
  Size start1 = mp;
  Size start2 = diag - mp;

  Size right_diag = min(n1 + n2, diag + work_per_thread);
  // XXX we could alternatively shuffle to find the right_mp
  Size right_mp = mgpu::MergePath<mgpu::MgpuBoundsLower>(first1, n1, first2, n2, right_diag, comp);
  Size end1 = right_mp;
  Size end2 = right_diag - right_mp;

  // each thread does a sequential merge
  thrust::merge(thrust::seq,
                first1 + start1, first1 + end1,
                first2 + start2, first2 + end2,
                result + work_per_thread * threadIdx.x,
                comp);
  __syncthreads();
}


// block-wise inplace merge for when we have a static bound on the size of the result (block_size * work_per_thread)
template<unsigned int block_size,
         unsigned int work_per_thread,
         typename Iterator,
         typename Size,
         typename Compare>
__device__ void bounded_inplace_merge(Iterator first, Size n1, Size n2, Compare comp)
{
  Iterator first2 = first + n1;

  // don't ask for an out-of-bounds diagonal
  Size diag = min(n1 + n2, work_per_thread * threadIdx.x);

  Size mp = mgpu::MergePath<mgpu::MgpuBoundsLower>(first, n1, first2, n2, diag, comp);

  // compute the ranges of the sources
  Size start1 = mp;
  Size start2 = diag - mp;

  Size end1 = n1;
  Size end2 = n2;
  
  // each thread does a local sequential merge
  typedef typename thrust::iterator_value<Iterator>::type value_type;
  value_type local_result[work_per_thread];
  sequential_bounded_merge<work_per_thread>(first  + start1, first  + end1,
                                            first2 + start2, first2 + end2,
                                            local_result, comp);

  __syncthreads();

  // store the result
  // XXX we unconditionally copy work_per_thread elements here, even if input was partially-sized
  thrust::copy_n(thrust::seq, local_result, work_per_thread, first + work_per_thread * threadIdx.x);
  __syncthreads();
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


// staged, block-wise merge for when we have a static bound on the size of the result (block_size * work_per_thread)
template<unsigned int block_size,
         unsigned int work_per_thread,
         typename Iterator1, typename Size1,
         typename Iterator2, typename Size2,
         typename Iterator3,
	 typename Compare>
__device__
void staged_bounded_merge(Iterator1 first1, Size1 n1,
                          Iterator2 first2, Size2 n2,
                          Iterator3 result,
                          Compare comp)
{
  typedef typename thrust::iterator_value<Iterator3>::type value_type;

  using thrust::system::cuda::detail::detail::uninitialized_array;
  __shared__ uninitialized_array<value_type, block_size * (work_per_thread + 1)> s_keys;

  // stage the input through shared memory.
  // XXX replacing copy_n_fast with copy_n results in a 10% performance hit
  block::copy_n_fast<block_size, work_per_thread>(first1, n1, s_keys.begin());
  block::copy_n_fast<block_size, work_per_thread>(first2, n2, s_keys.begin() + n1);
  __syncthreads();

  // cooperatively merge in place
  block::bounded_inplace_merge<block_size, work_per_thread>(s_keys.begin(), n1, n2, comp);
  
  // store result in smem to result
  block::copy_n<block_size>(s_keys.begin(), n1 + n2, result);
  __syncthreads();
}


} // end block


// Returns (start1, end1, start2, end2) into mergesort input lists between mp0 and mp1.
__host__ __device__
thrust::tuple<int,int,int,int> find_mergesort_interval(int3 frame, int num_blocks_per_merge, int block_idx, int num_elements_per_block, int n, int mp, int right_mp)
{
  // Locate diag from the start of the A sublist.
  int diag = num_elements_per_block * block_idx - frame.x;
  int start1 = frame.x + mp;
  int end1 = min(n, frame.x + right_mp);
  int start2 = min(n, frame.y + diag - mp);
  int end2 = min(n, frame.y + diag + num_elements_per_block - right_mp);
  
  // The end partition of the last block for each merge operation is computed
  // and stored as the begin partition for the subsequent merge. i.e. it is
  // the same partition but in the wrong coordinate system, so its 0 when it
  // should be listSize. Correct that by checking if this is the last block
  // in this merge operation.
  if(num_blocks_per_merge - 1 == ((num_blocks_per_merge - 1) & block_idx))
  {
    end1 = min(n, frame.x + frame.z);
    end2 = min(n, frame.y + frame.z);
  }

  return thrust::make_tuple(start1, end1, start2, end2);
}


__host__ __device__
thrust::tuple<int,int,int,int> locate_merge_partitions(int n, int block_idx, int num_blocks_per_merge, int num_elements_per_block, int mp, int right_mp)
{
  int first_block_in_partition = ~(num_blocks_per_merge - 1) & block_idx;
  int size = num_elements_per_block * (num_blocks_per_merge >> 1);
  int3 frame = make_int3(num_elements_per_block * first_block_in_partition, num_elements_per_block * first_block_in_partition + size, size);

  return find_mergesort_interval(frame, num_blocks_per_merge, block_idx, num_elements_per_block, n, mp, right_mp);
}


template<unsigned int block_size,
         unsigned int work_per_thread,
         typename Size,
         typename Iterator1,
         typename Iterator2,
         typename Iterator3,
         typename Compare>
__global__
void merge_adjacent_partitions(Size num_blocks_per_merge,
                               Iterator1 first, Size n,
                               Iterator2 merge_paths,
                               Iterator3 result,
                               Compare comp)
{
  const int work_per_block = block_size * work_per_thread;
  
  int start1 = 0, end1 = 0, start2 = 0, end2 = 0;

  thrust::tie(start1,end1,start2,end2) =
    locate_merge_partitions(n, blockIdx.x, num_blocks_per_merge, work_per_block, merge_paths[blockIdx.x], merge_paths[blockIdx.x + 1]);
  
  block::staged_bounded_merge<block_size, work_per_thread>(first + start1, end1 - start1,
                                                           first + start2, end2 - start2,
                                                           result + blockIdx.x * work_per_block,
                                                           comp);
}


template<typename Iterator, typename Size, typename Compare>
struct locate_merge_path
{
  Iterator haystack_first;
  Size haystack_size;
  Size num_elements_per_block;
  Size num_blocks_per_merge;
  Compare comp;

  locate_merge_path(Iterator haystack_first, Size haystack_size, Size num_elements_per_block, Size num_blocks_per_merge, Compare comp)
    : haystack_first(haystack_first),
      haystack_size(haystack_size),
      num_elements_per_block(num_elements_per_block),
      num_blocks_per_merge(num_blocks_per_merge),
      comp(comp)
  {}

  template<typename Index>
  __host__ __device__
  Index operator()(Index merge_path_idx)
  {
    Size a0 = 0, b0 = 0;
    Size gid = num_elements_per_block * merge_path_idx;

    Size first_block_in_partition = ~(num_blocks_per_merge - 1) & merge_path_idx;
    Size size = num_elements_per_block * (num_blocks_per_merge >> 1);
    thrust::tuple<Size,Size,Size> frame =
      thrust::make_tuple(num_elements_per_block * first_block_in_partition, num_elements_per_block * first_block_in_partition + size, size);

    a0 = thrust::get<0>(frame);
    b0 = thrust::min(haystack_size, thrust::get<1>(frame));
    Size n2 = thrust::min(haystack_size, thrust::get<1>(frame) + size) - b0;
    Size n1 = thrust::min(haystack_size, thrust::get<0>(frame) + size) - a0;
    
    // Put the cross-diagonal into the coordinate system of the input
    // lists.
    gid -= a0;

    return mgpu::MergePath<mgpu::MgpuBoundsLower>(haystack_first + a0, n1, haystack_first + b0, n2, min(gid, n1 + n2), comp);
  }
};


template<typename Iterator1, typename Size, typename Iterator2, typename Compare>
void locate_merge_paths(Iterator1 result, Size n, Iterator2 haystack_first, Size haystack_size, Size num_elements_per_block, Size num_blocks_per_merge, Compare comp)
{
  locate_merge_path<Iterator2,Size,Compare> f(haystack_first, haystack_size, num_elements_per_block, num_blocks_per_merge, comp);

  thrust::tabulate(thrust::cuda::par, result, result + n, f);
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
  
  const int work_per_block = block_size * work_per_thread;

  int numBlocks = MGPU_DIV_UP(n, work_per_block);
  int numPasses = mgpu::FindLog2(numBlocks, true);
  
  MGPU_MEM(T) destDevice = exec.ctx->Malloc<T>(n);
  T* source = thrust::raw_pointer_cast(&*first);
  T* dest = destDevice->get();
  
  mgpu::KernelBlocksort<Tuning, false>
    <<<numBlocks, launch.x, 0>>>(source, (const int*)0,
    n, (1 & numPasses) ? dest : source, (int*)0, comp);
  if(1 & numPasses) std::swap(source, dest);

  MGPU_MEM(int) merge_paths = exec.ctx->Malloc<T>(numBlocks + 1);
  
  for(int pass = 0; pass < numPasses; ++pass)
  {
    int num_blocks_per_merge = 2 << pass;
    locate_merge_paths(merge_paths->get(), numBlocks + 1, source, n, work_per_block, num_blocks_per_merge, comp);
    
    merge_adjacent_partitions<block_size, work_per_thread><<<numBlocks, launch.x>>>(num_blocks_per_merge, source, n, merge_paths->get(), dest, comp);

    std::swap(dest, source);
  }
}


} // end stable_merge_sort_detail2


template<typename RandomAccessIterator, typename Compare>
void stable_merge_sort(my_policy &exec,
                       RandomAccessIterator first,
                       RandomAccessIterator last,
                       Compare comp)
{
  stable_merge_sort_detail2::stable_merge_sort(exec, first, last, comp);
}

