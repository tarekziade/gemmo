#include "gemmology/gemmology.h"
#include "xsimd/xsimd.hpp"
#include <atomic>
#include <cassert>
#include <chrono> // For timing
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <iomanip> // For formatted output
#include <iostream>
#include <queue>
#include <thread>
#include <vector>
#include <random>
#include <algorithm>


#if defined(NEON_I8MM)
using arch = xsimd::i8mm<xsimd::neon64>;
#elif defined(NEON)
using arch = xsimd::neon64; 
#else
using arch = xsimd::default_arch;
#endif

using vuint8_t = xsimd::batch<uint8_t, arch>;
using vint8_t = xsimd::batch<int8_t, arch>;
using vint16_t = xsimd::batch<int16_t, arch>;
using vint32_t = xsimd::batch<int32_t, arch>;
using vuint32_t = xsimd::batch<uint32_t, arch>;

// Simple thread pool class
class ThreadPool {
public:
  ThreadPool(size_t num_threads) : active_tasks(0) {
    for (size_t i = 0; i < num_threads; ++i) {
      workers.emplace_back([this]() {
        while (true) {
          std::function<void()> task;

          {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->condition.wait(
                lock, [this]() { return this->stop || !this->tasks.empty(); });

            if (this->stop && this->tasks.empty())
              return;

            task = std::move(this->tasks.front());
            this->tasks.pop();
            ++active_tasks;
          }

          task();

          {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            --active_tasks;
            if (tasks.empty() && active_tasks == 0) {
              completion_condition.notify_all();
            }
          }
        }
      });
    }
  }

  template <class F> void enqueue(F &&f) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      tasks.emplace(std::forward<F>(f));
    }
    condition.notify_one();
  }

  void wait() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    completion_condition.wait(
        lock, [this]() { return tasks.empty() && active_tasks == 0; });
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers)
      worker.join();
  }

private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex queue_mutex;
  std::condition_variable condition;
  std::condition_variable completion_condition;
  std::atomic<int> active_tasks;
  bool stop = false;
};

// Global thread pool
static ThreadPool threadPool(4); // Initialize thread pool with 4 threads

void GemmMatMulM1(const uint8_t *inputMatrixA, const int8_t *inputMatrixB,
                size_t rowsA, size_t width, size_t colsB, uint8_t zeroPointA,
                const uint8_t *zeroPointB, const float *b_scale_data,
                bool is_b_scale_per_column, float *output) {

  vint32_t vzeroPointA = zeroPointA;

  for (size_t col_idx = 0; col_idx < colsB; col_idx += 4 * vint32_t::size) {
      vint32_t vtemp_result[4] = {}; // this stores a * b
      vint32_t vadj[4] = {}; // this stores zeroPointA * b
      for (size_t k = 0; k < width; k+=4) {
        vuint8_t va_data = xsimd::bitwise_cast<uint8_t>(vint32_t(*(int32_t*)&inputMatrixA[k+0]));

        vint8_t vb_data0 = vint8_t::load_unaligned(&inputMatrixB[(k+0) * colsB + col_idx +  0]);
        vint8_t vb_data1 = vint8_t::load_unaligned(&inputMatrixB[(k+1) * colsB + col_idx +  0]);
        vint8_t vb_data2 = vint8_t::load_unaligned(&inputMatrixB[(k+2) * colsB + col_idx +  0]);
        vint8_t vb_data3 = vint8_t::load_unaligned(&inputMatrixB[(k+3) * colsB + col_idx +  0]);

        vint16_t vb_data_lo0 = xsimd::bit_cast<vint16_t>(zip_lo(vb_data0, vb_data1)); // 0,0 1,0 0,1 1,1 0,2 1,2
        vint16_t vb_data_lo1 = xsimd::bit_cast<vint16_t>(zip_lo(vb_data2, vb_data3)); // 2,0 3,0 2,1 3,1 2,2 3,2
        auto vb_datap0 = xsimd::bit_cast<vint8_t>(zip_lo(vb_data_lo0, vb_data_lo1));
        auto vb_datap1 = xsimd::bit_cast<vint8_t>(zip_hi(vb_data_lo0, vb_data_lo1));

        vtemp_result[0] = gemmology::maddw(va_data, vb_datap0, vtemp_result[0]);
        vtemp_result[1] = gemmology::maddw(va_data, vb_datap1, vtemp_result[1]);

        vadj[0] = gemmology::maddw(vuint8_t(1), vb_datap0, vadj[0]);
        vadj[1] = gemmology::maddw(vuint8_t(1), vb_datap1, vadj[1]);

        vint16_t vb_data_hi0 = xsimd::bit_cast<vint16_t>(zip_hi(vb_data0, vb_data1)); // 0,7 1,7 0,8 1,8 0,9 1,9
        vint16_t vb_data_hi1 = xsimd::bit_cast<vint16_t>(zip_hi(vb_data2, vb_data3)); // 2,7 3,7 2,8 3,8 2,9 3,9

        auto vb_datap2 = xsimd::bit_cast<vint8_t>(zip_lo(vb_data_hi0, vb_data_hi1));
        auto vb_datap3 = xsimd::bit_cast<vint8_t>(zip_hi(vb_data_hi0, vb_data_hi1));

        vtemp_result[2] = gemmology::maddw(va_data, vb_datap2, vtemp_result[2]);
        vtemp_result[3] = gemmology::maddw(va_data, vb_datap3, vtemp_result[3]);

        vadj[2] = gemmology::maddw(vuint8_t(1), vb_datap2, vadj[2]);
        vadj[3] = gemmology::maddw(vuint8_t(1), vb_datap3, vadj[3]);
      }

      (vtemp_result[0] - vzeroPointA * vadj[0]).store_unaligned(&output[col_idx + 0 * vint32_t::size]);
      (vtemp_result[1] - vzeroPointA * vadj[1]).store_unaligned(&output[col_idx + 1 * vint32_t::size]);
      (vtemp_result[2] - vzeroPointA * vadj[2]).store_unaligned(&output[col_idx + 2 * vint32_t::size]);
      (vtemp_result[3] - vzeroPointA * vadj[3]).store_unaligned(&output[col_idx + 3 * vint32_t::size]);
  }


  int32_t a_acc = 0;
  for (size_t k = 0; k < width; k+=1) {
      a_acc +=  inputMatrixA[k];
  }
  for (size_t col_idx = 0; col_idx < colsB; col_idx += 1) {
      output[col_idx] += - a_acc * zeroPointB[col_idx];
  }

  for (size_t col_idx = 0; col_idx < colsB; col_idx += 1) {
      output[col_idx] += zeroPointA * zeroPointB[col_idx] * width;
  }

  // probably want a better spot for this
  for (size_t col_idx = 0; col_idx < colsB; ++col_idx) {
    if (is_b_scale_per_column) {
      output[col_idx] *= b_scale_data[col_idx];
    } else {
      output[col_idx] *= b_scale_data[0];
    }
  }

}

/**
 * Naive implementation
 */
void NaiveMatMul(const uint8_t *inputMatrixA, const int8_t *inputMatrixB,
                 size_t rowsA, size_t width, size_t colsB, uint8_t zeroPointA,
                 const uint8_t *zeroPointB, const float *b_scale_data,
                 bool is_b_scale_per_column, float *output) {

  for (size_t rowIndex = 0; rowIndex < rowsA; ++rowIndex) {
    const uint8_t *aRow = inputMatrixA + rowIndex * width; // Start of row in A
    for (size_t colIndex = 0; colIndex < colsB; ++colIndex) {
      int32_t tempResult = 0;

      for (size_t k = 0; k < width; ++k) {
        // Row-major access
        uint8_t aValue = aRow[k];

        // Column-major access for B
        int8_t bValue = inputMatrixB[k * colsB + colIndex];

        // Adjust for zero-point offsets
        int32_t adjustedA =
            static_cast<int32_t>(aValue) - static_cast<int32_t>(zeroPointA);
        int32_t adjustedB = static_cast<int32_t>(bValue);

        if (is_b_scale_per_column) {
          adjustedB -= static_cast<int32_t>(zeroPointB[colIndex]);
        } else {
          adjustedB -= static_cast<int32_t>(zeroPointB[0]);
        }
        // Accumulate product
        tempResult += adjustedA * adjustedB;
      }

      float scaledResult = tempResult;
      if (is_b_scale_per_column) {
        scaledResult *= b_scale_data[colIndex];
      } else {
        scaledResult *= b_scale_data[0];
      }

      // Store the scaled result in y_data
      output[rowIndex * colsB + colIndex] = scaledResult;
    }
  }
}

/**
 * Gemmology implementation
 */
void GemmMatMul(const uint8_t *inputMatrixA, const int8_t *inputMatrixB,
                size_t rowsA, size_t width, size_t colsB, uint8_t zeroPointA,
                const uint8_t *zeroPointB, const float *b_scale_data,
                bool is_b_scale_per_column, float *output) {
  if ( rowsA == 1 )
    return GemmMatMulM1(inputMatrixA, inputMatrixB,
        rowsA, width, colsB, zeroPointA,
        zeroPointB, b_scale_data,
        is_b_scale_per_column, output
        );

  int8_t *transposedB;
  posix_memalign((void**)&transposedB, 64, width * colsB);
  for (size_t k = 0; k < width; k += 4) {
    for (size_t colIndex = 0; colIndex < colsB; colIndex += 4 * vint32_t::size) {
      vint8_t vinputMatrixB0 = vint8_t::load_unaligned(&inputMatrixB[(k + 0) * colsB + colIndex]);
      vint8_t vinputMatrixB1 = vint8_t::load_unaligned(&inputMatrixB[(k + 1) * colsB + colIndex]);
      vint8_t vinputMatrixB2 = vint8_t::load_unaligned(&inputMatrixB[(k + 2) * colsB + colIndex]);
      vint8_t vinputMatrixB3 = vint8_t::load_unaligned(&inputMatrixB[(k + 3) * colsB + colIndex]);

      vint16_t vinputMatrixB_lo0 = xsimd::bit_cast<vint16_t>(zip_lo(vinputMatrixB0, vinputMatrixB1));
      vint16_t vinputMatrixB_lo1 = xsimd::bit_cast<vint16_t>(zip_lo(vinputMatrixB2, vinputMatrixB3));

      vint16_t vinputMatrixB_hi0 = xsimd::bit_cast<vint16_t>(zip_hi(vinputMatrixB0, vinputMatrixB1));
      vint16_t vinputMatrixB_hi1 = xsimd::bit_cast<vint16_t>(zip_hi(vinputMatrixB2, vinputMatrixB3));

      xsimd::bit_cast<vint8_t>(zip_lo(vinputMatrixB_lo0, vinputMatrixB_lo1)).store_unaligned(&transposedB[(k+0) * colsB + colIndex]);
      xsimd::bit_cast<vint8_t>(zip_hi(vinputMatrixB_lo0, vinputMatrixB_lo1)).store_unaligned(&transposedB[(k+1) * colsB + colIndex]);
      xsimd::bit_cast<vint8_t>(zip_lo(vinputMatrixB_hi0, vinputMatrixB_hi1)).store_unaligned(&transposedB[(k+2) * colsB + colIndex]);
      xsimd::bit_cast<vint8_t>(zip_hi(vinputMatrixB_hi0, vinputMatrixB_hi1)).store_unaligned(&transposedB[(k+3) * colsB + colIndex]);
    }
  }

  vint32_t vzeroPointA = zeroPointA;
  for (size_t rowIndex = 0; rowIndex < rowsA; ++rowIndex) {
    for (size_t colIndex = 0; colIndex < colsB; colIndex += 4 * vint32_t::size) {
      vint32_t vacc[4] = {};
      vint32_t vadj[4] = {}; // this stores zeroPointA * b
      for (size_t k = 0; k < width; k += 4) {
        vuint8_t vinputMatrixA = xsimd::bitwise_cast<uint8_t>(vuint32_t(*(uint32_t*)(inputMatrixA + rowIndex * width + k)));
        vint8_t vtransposedB0 = vint8_t::load_unaligned(&transposedB[(k + 0) * colsB + colIndex]);
        vint8_t vtransposedB1 = vint8_t::load_unaligned(&transposedB[(k + 1) * colsB + colIndex]);
        vint8_t vtransposedB2 = vint8_t::load_unaligned(&transposedB[(k + 2) * colsB + colIndex]);
        vint8_t vtransposedB3 = vint8_t::load_unaligned(&transposedB[(k + 3) * colsB + colIndex]);

        vacc[0] = gemmology::maddw(vinputMatrixA, vtransposedB0, vacc[0]);
        vacc[1] = gemmology::maddw(vinputMatrixA, vtransposedB1, vacc[1]);
        vacc[2] = gemmology::maddw(vinputMatrixA, vtransposedB2, vacc[2]);
        vacc[3] = gemmology::maddw(vinputMatrixA, vtransposedB3, vacc[3]);

        vadj[0] = gemmology::maddw(vuint8_t(1), vtransposedB0, vadj[0]);
        vadj[1] = gemmology::maddw(vuint8_t(1), vtransposedB1, vadj[1]);
        vadj[2] = gemmology::maddw(vuint8_t(1), vtransposedB2, vadj[2]);
        vadj[3] = gemmology::maddw(vuint8_t(1), vtransposedB3, vadj[3]);
      }
      (vacc[0] - vzeroPointA * vadj[0]).store_aligned(&output[rowIndex * colsB + colIndex + 0 * vint32_t::size]);
      (vacc[1] - vzeroPointA * vadj[1]).store_aligned(&output[rowIndex * colsB + colIndex + 1 * vint32_t::size]);
      (vacc[2] - vzeroPointA * vadj[2]).store_aligned(&output[rowIndex * colsB + colIndex + 2 * vint32_t::size]);
      (vacc[3] - vzeroPointA * vadj[3]).store_aligned(&output[rowIndex * colsB + colIndex + 3 * vint32_t::size]);
    }
  }
  free(transposedB);

  int32_t a_acc = 0;
  for (size_t k = 0; k < width; k+=1) {
      a_acc +=  inputMatrixA[k];
  }

  for (size_t row_idx = 0; row_idx < rowsA; ++row_idx) {
    for (size_t col_idx = 0; col_idx < colsB; col_idx += 1) {
        output[row_idx * colsB + col_idx] += - a_acc * zeroPointB[col_idx];
        output[row_idx * colsB + col_idx] += zeroPointA * zeroPointB[col_idx] * width;
    }
  }

  // probably want a better spot for this
  for (size_t row_idx = 0; row_idx < rowsA; ++row_idx) {
    for (size_t col_idx = 0; col_idx < colsB; ++col_idx) {
      if (is_b_scale_per_column) {
        output[row_idx * colsB + col_idx] *= b_scale_data[col_idx];
      } else {
        output[row_idx * colsB + col_idx] *= b_scale_data[0];
      }
    }
  }
}

void CompareMatMul(size_t rowsA, size_t width, size_t colsB, uint8_t zeroPointA,
                   bool profile) {
  if (width % 16 != 0) {
    throw std::runtime_error(
        "Width must be divisible by 16 for SIMD operations.");
  }

  std::cout << "Comparing MatMul: " << rowsA << " x " << width << " x " << colsB << std::endl;

  uint8_t *zeroPointB = new uint8_t[colsB];
  for (size_t i = 0; i < colsB; ++i) {
    zeroPointB[i] = i;
  }
  const bool is_b_scale_per_column = true;

  uint8_t *inputMatrixA = new uint8_t[rowsA * width];
  // Matrix A: rowsA x width
  for (size_t row = 0; row < rowsA; ++row) {
    for (size_t col = 0; col < width; ++col) {
      inputMatrixA[row * width + col] = col % 256; // Fill with values 0 to 255
    }
  }

  int8_t *inputMatrixB = new int8_t[width * colsB];
  // Matrix B: width x colsB
  for (size_t row = 0; row < width; ++row) {
    for (size_t col = 0; col < colsB; ++col) {
      inputMatrixB[row * colsB + col] =
          (row * colsB + col) % 255 - 127; // Fill with values -127 to 127
    }
  }

  for (size_t row = 0; row < width; ++row) {
    for (size_t col = 0; col < colsB; ++col) {
      inputMatrixB[row * colsB + col] =
          row * colsB + col + 1; // Fill row by row
    }
  }

  float *b_scale_data = new float[colsB];
  for (size_t i = 0; i < colsB; ++i) {
    b_scale_data[i] = 0.000065f;
  }

  std::vector<float> output1(rowsA * colsB, 0);
  std::vector<float> output2(rowsA * colsB, 0);

  if (!profile) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int x = 0; x < (profile ? 100 : 1); ++x) {
      NaiveMatMul(inputMatrixA, inputMatrixB, rowsA, width, colsB, zeroPointA,
                  zeroPointB, b_scale_data, is_b_scale_per_column,

                  output1.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "NaiveMatMul took " << duration1 << " microseconds."
              << std::endl;
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int x = 0; x < (profile ? 1 : 1); ++x) {
    GemmMatMul(inputMatrixA, inputMatrixB, rowsA, width, colsB, zeroPointA,
               zeroPointB, b_scale_data, is_b_scale_per_column,

               output2.data());
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration2 =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  std::cout << "GemmMatMul took " << duration2 << " microseconds." << std::endl;
  if (!profile) {
    for (size_t i = 0; i < rowsA * colsB; ++i) {
      int8_t diff = output1[i] - output2[i];
      if (diff == 0) {
        continue;
      }
      std::cout << "Index " << i << ": NaiveMatMul=" << output1[i]
                << ", GemmMatMul=" << output2[i]
                << ", Difference=" << std::abs(output1[i] - output2[i])
                << std::endl;
      throw std::runtime_error("Different results!");
    }
    if (std::equal(output1.begin(), output1.end(), output2.begin(),
                   [](float a, float b) { return std::abs(a - b) < 1e-4; })) {
      std::cout << "Both implementations produce the same result!" << std::endl;
    } else {
      std::cout << "The implementations produce different results!"
                << std::endl;
    }
  }

  free(inputMatrixA);
  free(inputMatrixB);
}

int main(int argc, char **argv) {
  bool profile = false;
  if (argc > 1 && std::strcmp(argv[1], "--profile") == 0) {
    profile = true;
  }
  if (profile) {
    CompareMatMul(172, 1024, 1024, 123, profile);
    CompareMatMul(1, 1024, 1024, 123, profile);

  } else {
    CompareMatMul(172, 1024, 1024, 123, profile);
    CompareMatMul(1, 128, 256, 0, profile);
    CompareMatMul(1, 128, 256, 1, profile);
    // CompareMatMul(256, 256, 256, 0, profile);
    CompareMatMul(1, 1024, 1024, 0, profile);
    CompareMatMul(1, 1024, 1024, 123, profile);
    CompareMatMul(1, 1024, 4096, 123, profile);
  }

  return 0;
}
