#pragma once

template<typename Closure, typename RandomAccessIterator>
  struct virtualized_smem_closure
    : Closure
{
  typedef Closure super_t;

  size_t num_elements_per_block;
  RandomAccessIterator virtual_smem;

  virtualized_smem_closure(Closure closure, size_t num_elements_per_block, RandomAccessIterator virtual_smem)
    : super_t(closure),
      num_elements_per_block(num_elements_per_block),
      virtual_smem(virtual_smem)
  {}

  __device__ __thrust_forceinline__
  void operator()()
  {
    typename super_t::context_type ctx;

    RandomAccessIterator smem = virtual_smem + num_elements_per_block * ctx.block_index();

    super_t::operator()(smem);
  }
};

