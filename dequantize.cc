#include "xsimd/xsimd.hpp"
#include "gemmology/gemmology.h"
#if 0
__attribute__((noinline)) void DequantizeLinear(
    size_t M, size_t K, size_t N, const uint8_t* input,
    float* scale, float* output, const uint8_t* zero_point)
{
  using arch = xsimd::default_arch;
  using vint8_t = xsimd::batch<int8_t, arch>;
  using vuint8_t = xsimd::batch<uint8_t, arch>;
  using vuint16_t = xsimd::batch<uint16_t, arch>;
  using vint32_t = xsimd::batch<int32_t, arch>;
  using vfloat_t = xsimd::batch<float, arch>;

  for (size_t m = 0; m < M; m++) {
    for (size_t k = 0; k < K; k++) {
      vuint8_t vzp = zero_point ? static_cast<int32_t>(zero_point[k]) : 0;
      vint8_t comb = vint8_t{
        1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1,
        1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1,
      };

      vfloat_t vkscale = scale[k];

      for (size_t n = 0; n < N; n+= 4 * vint32_t::size) {
        vuint8_t vinput = vuint8_t::load_aligned(&input[m * K * N + k * N + n]);
        auto vtmp_lo = xsimd::zip_lo(vinput, vzp);
        auto vtmp_hi = xsimd::zip_hi(vinput, vzp);

        auto vtmp_lo0 = xsimd::zip_lo(vtmp_lo, vuint8_t(0));
        auto vtmp_lo1 = xsimd::zip_hi(vtmp_lo, vuint8_t(0));

        auto vtmp_hi0 = xsimd::zip_lo(vtmp_hi, vuint8_t(0));
        auto vtmp_hi1 = xsimd::zip_hi(vtmp_hi, vuint8_t(0));

        vint32_t vdiff_lo0 = gemmology::maddw(vtmp_lo0, comb, vint32_t(0));
        vfloat_t voutput_lo0 = xsimd::batch_cast<float>(vdiff_lo0) * vkscale;

        vint32_t vdiff_lo1 = gemmology::maddw(vtmp_lo1, comb, vint32_t(0));
        vfloat_t voutput_lo1 = xsimd::batch_cast<float>(vdiff_lo1) * vkscale;

        vint32_t vdiff_hi0 = gemmology::maddw(vtmp_hi0, comb, vint32_t(0));
        vfloat_t voutput_hi0 = xsimd::batch_cast<float>(vdiff_hi0) * vkscale;

        vint32_t vdiff_hi1 = gemmology::maddw(vtmp_hi1, comb, vint32_t(0));
        vfloat_t voutput_hi1 = xsimd::batch_cast<float>(vdiff_hi1) * vkscale;

        voutput_lo0.store_aligned(&output[m * K * N + k * N + n + 0 * vint32_t::size]);
        voutput_lo1.store_aligned(&output[m * K * N + k * N + n + 1 * vint32_t::size]);
        voutput_hi0.store_aligned(&output[m * K * N + k * N + n + 2 * vint32_t::size]);
        voutput_hi1.store_aligned(&output[m * K * N + k * N + n + 3 * vint32_t::size]);
      }
    }
  }
}
#endif

__attribute__((noinline)) void DequantizeLinearRef(
    size_t M, size_t K, size_t N, const uint8_t* input,
    float* scale, float* output, const uint8_t* zero_point)
{
  for (size_t m = 0; m < M; m++) {
    for (size_t k = 0; k < K; k++) {
      int32_t zp = zero_point ? static_cast<int32_t>(zero_point[k]) : 0;
      for (size_t n = 0; n < N; n+=1) {
        output[ m * K * N + k * N + n] = float(input[m * K * N + k * N + n] - zp) * scale[k];
      }
    }
  }
}

using arch = xsimd::neon; // xsimd::default_arch;
using vuint8_t = xsimd::batch<uint8_t, arch>;
using vint32_t = xsimd::batch<int32_t, arch>;
using vfloat_t = xsimd::batch<float, arch>;
vint32_t upcast(const uint8_t * input) {
  // The compiler knows how to optimize this away, yay
  alignas(arch::alignment()) int32_t input_buffer[vint32_t::size];
  for(int p = 0; p < vint32_t::size; ++p)
    input_buffer[p] = input[p];
  return vint32_t::load_aligned(input_buffer);
}

__attribute__((noinline)) void DequantizeLinear(
    size_t M, size_t K, size_t N, const uint8_t* input,
    float* scale, float* output, const uint8_t* zero_point)
{

  for (size_t m = 0; m < M; m++) {
    for (size_t k = 0; k < K; k++) {
      vint32_t vzp = zero_point ? static_cast<int32_t>(zero_point[k]) : 0;
      vfloat_t vkscale = scale[k];

      for (size_t n = 0; n < N; n += 4 * vint32_t::size) {
#if 1
        vuint8_t vinput = vuint8_t::load_aligned(&input[m * K * N + k * N + n]);
        auto vinputA = xsimd::zip_lo(vinput, vuint8_t(0));
        auto vinputB = xsimd::zip_hi(vinput, vuint8_t(0));
        vint32_t vinput0 = xsimd::bitwise_cast<int32_t>(zip_lo(vinputA, vuint8_t(0)));
        vint32_t vinput1 = xsimd::bitwise_cast<int32_t>(zip_hi(vinputA, vuint8_t(0)));
        vint32_t vinput2 = xsimd::bitwise_cast<int32_t>(zip_lo(vinputB, vuint8_t(0)));
        vint32_t vinput3 = xsimd::bitwise_cast<int32_t>(zip_hi(vinputB, vuint8_t(0)));
#else
        vint32_t vinput0 = upcast(&input[m * K * N + k * N + n + 0 * vint32_t::size]);
        vint32_t vinput1 = upcast(&input[m * K * N + k * N + n + 1 * vint32_t::size]);
        vint32_t vinput2 = upcast(&input[m * K * N + k * N + n + 2 * vint32_t::size]);
        vint32_t vinput3 = upcast(&input[m * K * N + k * N + n + 3 * vint32_t::size]);
#endif

        vfloat_t voutput0 = xsimd::batch_cast<float>(vinput0 - vzp) * vkscale;
        vfloat_t voutput1 = xsimd::batch_cast<float>(vinput1 - vzp) * vkscale;
        vfloat_t voutput2 = xsimd::batch_cast<float>(vinput2 - vzp) * vkscale;
        vfloat_t voutput3 = xsimd::batch_cast<float>(vinput3 - vzp) * vkscale;

        voutput0.store_aligned(&output[ m * K * N + k * N + n + 0 * vint32_t::size]);
        voutput1.store_aligned(&output[ m * K * N + k * N + n + 1 * vint32_t::size]);
        voutput2.store_aligned(&output[ m * K * N + k * N + n + 2 * vint32_t::size]);
        voutput3.store_aligned(&output[ m * K * N + k * N + n + 3 * vint32_t::size]);
      }
    }
  }
}

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <sys/time.h>

int main() {
  size_t M = 32, K = 16, N = 64;
  float* scale, *output, *output_ref;
  uint8_t *input, *zero_point;
  posix_memalign((void**)&scale, 64, K * sizeof(float));
  posix_memalign((void**)&input, 64, K * M * N * sizeof(float));
  posix_memalign((void**)&output, 64, K * M * N * sizeof(float));
  posix_memalign((void**)&output_ref, 64, K * M * N * sizeof(float));
  posix_memalign((void**)&zero_point, 64, K * sizeof(uint8_t));

  for(int i = 0; i < K; ++i) {
    zero_point[i] = i;
    scale[i] = i;
  }
  for(int i = 0; i < K*M*N; ++i) {
    input[i] = i;
  }

  constexpr int count = 10000;

  timeval ref_start, ref_stop;
  gettimeofday(&ref_start, nullptr);
  for(int i = 0; i < count ; ++i)
    DequantizeLinearRef(M, K, N, input, scale, output_ref, zero_point);
  gettimeofday(&ref_stop, nullptr);

  timeval my_start, my_stop;
  gettimeofday(&my_start, nullptr);
  for(int i = 0; i < count ; ++i)
    DequantizeLinear(M, K, N, input, scale, output, zero_point);
  gettimeofday(&my_stop, nullptr);

  printf("ref: %lf ms\n", (ref_stop.tv_sec - ref_start.tv_sec) * 1000. + (ref_stop.tv_usec - ref_start.tv_usec) / 1000.);
  printf("my : %lf ms\n", (my_stop.tv_sec - my_start.tv_sec) * 1000. + (my_stop.tv_usec - my_start.tv_usec) / 1000.);

  for(int i = 0; i < K*M*N;++i)
    if(output[i] != output_ref[i]) {
      std::cout << "(" << i << ") " << output_ref[i] << " != " << output[i] << "\n";
      return 1;
    }
  return 0;
}
