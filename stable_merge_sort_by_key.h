#pragma once

#include "stable_merge_sort.h"
#include <thrust/iterator/zip_iterator.h>
#include <thrust/detail/internal_functional.h>

template<typename DerivedPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2, typename Compare>
void stable_merge_sort_by_key(thrust::system::cuda::execution_policy<DerivedPolicy> &exec,
                              RandomAccessIterator1 keys_first,
                              RandomAccessIterator1 keys_last,
                              RandomAccessIterator2 values_first,
                              Compare comp)
{
  typedef thrust::tuple<RandomAccessIterator1,RandomAccessIterator2> iterator_tuple;
  typedef thrust::zip_iterator<iterator_tuple> zip_iterator;

  zip_iterator zipped_first = thrust::make_zip_iterator(thrust::make_tuple(keys_first, values_first));
  zip_iterator zipped_last = thrust::make_zip_iterator(thrust::make_tuple(keys_last, values_first));

  thrust::detail::compare_first<Compare> comp_first(comp);

  stable_merge_sort(exec, zipped_first, zipped_last, comp_first);
}

template<typename DerivedPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2, typename Compare>
void stable_merge_sort_by_key(const thrust::system::cuda::execution_policy<DerivedPolicy> &exec,
                              RandomAccessIterator1 keys_first,
                              RandomAccessIterator1 keys_last,
                              RandomAccessIterator2 values_first,
                              Compare comp)
{
  stable_merge_sort_by_key(const_cast<thrust::system::cuda::execution_policy<DerivedPolicy>&>(exec),
                           keys_first,
                           keys_last,
                           values_first,
                           comp);
}

