#pragma once


template<typename Integer>
__host__ __device__ __thrust_forceinline__
Integer clz(Integer x)
{
  // XXX optimize by lowering to intrinsics
  
  Integer num_non_sign_bits = std::numeric_limits<Integer>::digits;
  for(int i = num_non_sign_bits; i >= 0; --i)
  {
    if((1 << i) & x)
    {
      return num_non_sign_bits - i;
    }
  }

  return num_non_sign_bits + 1;
}


template<typename Integer>
__host__ __device__ __thrust_forceinline__
bool is_power_of_2(Integer x)
{
  return 0 == (x & (x - 1));
}


template<typename Integer>
__host__ __device__ __thrust_forceinline__
Integer log2(Integer x)
{
  return std::numeric_limits<Integer>::digits - clz(x);
}


template<typename Integer>
__host__ __device__ __thrust_forceinline__
Integer log2_ri(Integer x)
{
  Integer result = log2(x);

  // this is where we round up to the nearest log
  if(!is_power_of_2(x))
  {
    ++result;
  }

  return result;
}


template<typename Integer>
__host__ __device__ __thrust_forceinline__
bool is_odd(Integer x)
{
  return 1 & x;
}

