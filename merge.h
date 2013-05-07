#pragma once

#include <thrust/iterator/iterator_traits.h>


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


template<typename Size, typename Iterator1, typename Iterator2, typename Compare>
__device__
Size merge_path(Size pos, Iterator1 first1, Size n1, Iterator2 first2, Size n2, Compare comp)
{
  Size begin = (pos >= n2) ? (pos - n2) : Size(0);
  Size end = thrust::min<Size>(pos, n1);
  
  while(begin < end)
  {
    Size mid = (begin + end) >> 1;

    if(comp(first2[pos - 1 - mid], first1[mid]))
    {
      end = mid;
    }
    else
    {
      begin = mid + 1;
    }
  }
  return begin;
}

