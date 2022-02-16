// circle -std=c++20 --cuda-path=/usr/local/cuda -sm_60 --verbose main.cpp -L/usr/local/cuda/lib64 -lcudart -o a.out

#include <chrono>
#include <functional>
#include <numeric>
#include <vector>
#include <iostream>
#include <random>
#include <sm_30_intrinsics.hpp>


template<class T>
struct managed_allocator
{
  using value_type = T;

  T* allocate(std::size_t n)
  {
    T* result{};
    cudaMallocManaged(reinterpret_cast<void**>(&result), sizeof(T) * n);
    return result;
  }

  void deallocate(T* ptr, std::size_t)
  {
    cudaFree(ptr);
  }
};


template<class T>
using device_vector = std::vector<T, managed_allocator<T>>;


template<class T>
concept plain_old_data = std::is_trivial_v<T> and std::is_standard_layout_v<T>;


template<std::integral I>
constexpr I ceil_div(I numerator, I denominator)
{
  return (numerator + denominator - I{1}) / denominator;
}


constexpr int warp_size = 32;


template<std::integral I>
constexpr bool is_pow2(I x)
{
  return 0 == (x & (x - 1));
}


template<plain_old_data T>
T shuffle_down(const T& x, int offset, int width)
{ 
  constexpr std::size_t num_words = ceil_div(sizeof(T), sizeof(int));

  union
  {
    int words[num_words];
    T value;
  } u;
  u.value = x;

  for(int i = 0; i < num_words; ++i)
  {
    u.words[i] = __shfl_down_sync(__activemask(), u.words[i], offset, width);
  }

  return u.value;
}


// the result is returned only to lane 0
// the result is undefined for other lanes
template<int num_threads = warp_size, plain_old_data T, std::invocable<T,T> F>
  requires (num_threads <= 32 and is_pow2(num_threads))
T warp_reduce(int lane, T value, int count, F binary_op)
{
  constexpr int num_passes = std::log2(num_threads);

  if(count == num_threads)
  {
    for(int pass = 0; pass < num_passes; ++pass)
    {
      int offset = 1 << pass;
      T other = shuffle_down(value, offset, num_threads);
      value = binary_op(value, other);
    }
  }
  else
  {
    for(int pass = 0; pass < num_passes; ++pass)
    {
      int offset = 1 << pass;
      T other = shuffle_down(value, offset, num_threads);
      if(lane + offset < count)
      {
        value = binary_op(value, other);
      }
    }
  }

  return value;
}


// the result is returned only to threadIdx.x == 0
// the result is undefined for other lanes
template<int block_size, plain_old_data T, std::invocable<T,T> F>
  requires (block_size % warp_size == 0)
T block_reduce(T value, int count, F binary_op)
{
  constexpr int num_warps = ceil_div(block_size, warp_size);

  __shared__ T s_partial_results[num_warps];

  int warp_idx = threadIdx.x / warp_size;
  int lane_idx = threadIdx.x % warp_size;

  int warp_count = warp_size;
  if((warp_idx + 1) * warp_size > count)
  {
    warp_count = count - warp_idx * warp_size;
  }

  // each warp computes a partial result
  value = warp_reduce(lane_idx, value, warp_count, binary_op);

  __syncthreads();

  // the warp's first lane stores its partial result
  if(lane_idx == 0 and warp_count > 0)
  {
    s_partial_results[warp_idx] = value;
  }

  __syncthreads();

  // the first warp computes the final result from the partials
  int num_partial_results = ceil_div(count, warp_size);
  if(warp_idx == 0)
  {
    value = (threadIdx.x < num_partial_results) ? s_partial_results[threadIdx.x] : value;
    value = warp_reduce(lane_idx, value, num_partial_results, binary_op);
  }

  return value;
}


template<int block_size, class BinaryOperation>
__global__ void reduce_tiles_kernel(const int* input, int tile_size, int size_of_final_tile, int* output, BinaryOperation binary_op)
{
  int this_tile_begin = tile_size * blockIdx.x;
  int this_tile_size = (blockIdx.x == gridDim.x - 1) ? size_of_final_tile : tile_size;

  // reduce this_tile_size inputs to <= block_size inputs
  int sum;
  int i = threadIdx.x;
  if(i < this_tile_size) sum = input[this_tile_begin + i];
  for(i += block_size; i < this_tile_size; i += block_size)
  {
    sum = binary_op(sum, input[this_tile_begin + i]);
  }

  // reduce across the block
  sum = block_reduce<block_size>(sum, std::min(this_tile_size, block_size), binary_op);
  
  if(threadIdx.x == 0)
  {
    output[blockIdx.x] = sum;
  }
}


template<int block_size, class BinaryOperation>
void plain_old_reduce(const int* input, int input_size, int* result, int* partial_sums, BinaryOperation op)
{
  // 80 is the number of SMs on Titan V
  // 695 was determined empirically
  int max_num_ctas = 80 * 695;

  // 11 was revealed to me in a dream
  int work_per_thread = 11;

  int min_tile_size = block_size * work_per_thread;

  int tile_size = ceil_div(input_size, max_num_ctas);
  tile_size = std::max(tile_size, min_tile_size);

  int num_tiles = ceil_div(input_size, tile_size);

  int size_of_final_tile = (input_size % tile_size) ? (input_size % tile_size) : tile_size;

  if(num_tiles > 1)
  {
    reduce_tiles_kernel<block_size><<<num_tiles, block_size>>>(input, tile_size, size_of_final_tile, partial_sums, op);

    // finish up in a second phase by reducing a single tile with a larger block
    constexpr int block_size = 512;
    reduce_tiles_kernel<block_size><<<1, block_size>>>(partial_sums, num_tiles, num_tiles, result, op);
  }
  else
  {
    // the input is small enough that it only requires a single phase 
    constexpr int block_size = 512;
    reduce_tiles_kernel<512><<<1, block_size>>>(input, tile_size, size_of_final_tile, result, op);
  }
}


void test_correctness(int max_size)
{
  device_vector<int> input(max_size, 1);

  std::default_random_engine rng;
  for(int& x : input)
  {
    x = rng();
  }

  device_vector<int> result(1,0);
  device_vector<int> temporary(max_size);

  @meta for(int num_warps = 1; num_warps <= 32; ++num_warps)
  {{
    constexpr int block_size = num_warps * warp_size;

    std::cout << "Testing block_size = " << block_size << std::endl;

    for(int size = 1000; size < max_size; size += size / 100)
    {
      plain_old_reduce<block_size>(input.data(), size, result.data(), temporary.data(), std::plus{});
      cudaStreamSynchronize(0);

      // host reduce using std::accumulate.
      int ref = std::accumulate(input.begin(), input.begin() + size, 0);

      if(result[0] != ref)
      {
        printf("reduce:           %d\n", result[0]);
        printf("std::accumulate:  %d\n", ref);
        printf("Error at size: %d\n", size);
        exit(1);
      }
    }
  }}
}


double test_performance(int size, int num_trials)
{
  device_vector<int> input(size, 1);
  device_vector<int> result(1);
  device_vector<int> temporary(size);

  using namespace std::chrono;

  cudaDeviceSynchronize();

  auto start = system_clock::now();

  for(std::size_t i = 0; i < num_trials; ++i)
  {
    plain_old_reduce<32*6>(input.data(), size, result.data(), temporary.data(), std::plus{});
  };

  cudaDeviceSynchronize();

  auto elapsed = system_clock::now() - start;

  auto usecs = duration_cast<microseconds>(elapsed).count();

  double total_seconds = double(usecs) / 1000000;

  double mean_seconds = total_seconds / num_trials;

  std::size_t num_bytes = sizeof(int) * input.size();

  double bytes_per_second = double(num_bytes) / mean_seconds;

  double gigabytes_per_second = bytes_per_second / 1000000000;

  return gigabytes_per_second;
}


int main(int argc, char** argv)
{
  std::cout << "Testing correctness... " << std::endl;
  test_correctness(23456789);
  std::cout << "Done." << std::endl;

  std::cout << "Testing performance... " << std::flush;
  double bandwidth = test_performance(1 << 30, 1000);
  std::cout << "Done." << std::endl;

  std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

  return 0; 
}

