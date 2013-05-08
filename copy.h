#pragma once


namespace block
{


template<typename Context, typename Iterator1, typename Size, typename Iterator2>
__device__
void async_copy_n(Context &ctx, Iterator1 first, Size n, Iterator2 result)
{
  for(Size i = ctx.thread_index(); i < n; i += ctx.block_dimension())
  {
    result[i] = first[i];
  }
}


template<typename Context, typename Iterator1, typename Size, typename Iterator2>
__device__
void copy_n(Context &ctx, Iterator1 first, Size n, Iterator2 result)
{
  async_copy_n(ctx, first, n, result);
  ctx.barrier();
}


template<unsigned int work_per_thread, typename Context, typename Iterator1, typename Size, typename Iterator2>
__device__
void async_copy_n_global_to_shared(Context &ctx, Iterator1 first, Size n, Iterator2 result)
{
  typedef typename thrust::iterator_value<Iterator1>::type value_type;

  // stage copy through registers
  value_type reg[work_per_thread];

  // avoid conditional accesses when possible
  if(n >= ctx.block_dimension() * work_per_thread)
  {
    for(unsigned int i = 0; i < work_per_thread; ++i)
    {
      unsigned int idx = ctx.block_dimension() * i + ctx.thread_index();

      reg[i] = first[idx];
    }
  }
  else
  {
    for(unsigned int i = 0; i < work_per_thread; ++i)
    {
      unsigned int idx = ctx.block_dimension() * i + ctx.thread_index();

      if(idx < n) reg[i] = first[idx];
    }
  }

  // avoid conditional accesses when possible
  if(n >= ctx.block_dimension() * work_per_thread)
  {
    for(unsigned int i = 0; i < work_per_thread; ++i)
    {
      unsigned int idx = ctx.block_dimension() * i + ctx.thread_index();

      result[idx] = reg[i];
    }
  }
  else
  {
    for(unsigned int i = 0; i < work_per_thread; ++i)
    {
      unsigned int idx = ctx.block_dimension() * i + ctx.thread_index();

      if(idx < n) result[idx] = reg[i];
    }
  }
}


template<unsigned int work_per_thread, typename Context, typename Iterator1, typename Size, typename Iterator2>
__device__
void copy_n_global_to_shared(Context &ctx, Iterator1 first, Size n, Iterator2 result)
{
  async_copy_n_global_to_shared<work_per_thread>(ctx, first, n, result);
  ctx.barrier();
}


} // end block

