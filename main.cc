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

  vuint8_t bitMask = 0x80;

  for (size_t col_idx = 0; col_idx < colsB; col_idx += 4 * vint32_t::size) {
      vint32_t vtemp_result[4] = {}; // this stores a * b
      vint32_t vadj[4] = {}; // this stores zeroPointA * b
      for (size_t k = 0; k < width; k+=4) {
        vuint8_t va_data = xsimd::bitwise_cast<uint8_t>(vint32_t(*(int32_t*)&inputMatrixA[k+0]));
        vuint8_t va_data_sign = va_data & bitMask;
        vuint8_t va_data_trim = va_data & ~bitMask;

        vint8_t vb_data0 = vint8_t::load_unaligned(&inputMatrixB[(k+0) * colsB + col_idx +  0]);
        vint8_t vb_data1 = vint8_t::load_unaligned(&inputMatrixB[(k+1) * colsB + col_idx +  0]);
        vint8_t vb_data2 = vint8_t::load_unaligned(&inputMatrixB[(k+2) * colsB + col_idx +  0]);
        vint8_t vb_data3 = vint8_t::load_unaligned(&inputMatrixB[(k+3) * colsB + col_idx +  0]);

        vint16_t vb_data_lo0 = xsimd::bit_cast<vint16_t>(zip_lo(vb_data0, vb_data1)); // 0,0 1,0 0,1 1,1 0,2 1,2
        vint16_t vb_data_lo1 = xsimd::bit_cast<vint16_t>(zip_lo(vb_data2, vb_data3)); // 2,0 3,0 2,1 3,1 2,2 3,2
        vtemp_result[0] = gemmology::maddw(va_data_sign, xsimd::bit_cast<vint8_t>(zip_lo(vb_data_lo0, vb_data_lo1)), vtemp_result[0]);
        vtemp_result[0] = gemmology::maddw(va_data_trim, xsimd::bit_cast<vint8_t>(zip_lo(vb_data_lo0, vb_data_lo1)), vtemp_result[0]);
        vtemp_result[1] = gemmology::maddw(va_data_sign, xsimd::bit_cast<vint8_t>(zip_hi(vb_data_lo0, vb_data_lo1)), vtemp_result[1]);
        vtemp_result[1] = gemmology::maddw(va_data_trim, xsimd::bit_cast<vint8_t>(zip_hi(vb_data_lo0, vb_data_lo1)), vtemp_result[1]);

        vadj[0] = gemmology::maddw(vuint8_t(1), xsimd::bit_cast<vint8_t>(zip_lo(vb_data_lo0, vb_data_lo1)), vadj[0]);
        vadj[1] = gemmology::maddw(vuint8_t(1), xsimd::bit_cast<vint8_t>(zip_hi(vb_data_lo0, vb_data_lo1)), vadj[1]);

        vint16_t vb_data_hi0 = xsimd::bit_cast<vint16_t>(zip_hi(vb_data0, vb_data1)); // 0,7 1,7 0,8 1,8 0,9 1,9
        vint16_t vb_data_hi1 = xsimd::bit_cast<vint16_t>(zip_hi(vb_data2, vb_data3)); // 2,7 3,7 2,8 3,8 2,9 3,9
        vtemp_result[2] = gemmology::maddw(va_data_sign, xsimd::bit_cast<vint8_t>(zip_lo(vb_data_hi0, vb_data_hi1)), vtemp_result[2]);
        vtemp_result[2] = gemmology::maddw(va_data_trim, xsimd::bit_cast<vint8_t>(zip_lo(vb_data_hi0, vb_data_hi1)), vtemp_result[2]);
        vtemp_result[3] = gemmology::maddw(va_data_sign, xsimd::bit_cast<vint8_t>(zip_hi(vb_data_hi0, vb_data_hi1)), vtemp_result[3]);
        vtemp_result[3] = gemmology::maddw(va_data_trim, xsimd::bit_cast<vint8_t>(zip_hi(vb_data_hi0, vb_data_hi1)), vtemp_result[3]);

        vadj[2] = gemmology::maddw(vuint8_t(1), xsimd::bit_cast<vint8_t>(zip_lo(vb_data_hi0, vb_data_hi1)), vadj[2]);
        vadj[3] = gemmology::maddw(vuint8_t(1), xsimd::bit_cast<vint8_t>(zip_hi(vb_data_hi0, vb_data_hi1)), vadj[3]);
      }

      (vtemp_result[0] - vzeroPointA * vadj[0]).store_unaligned(&output[col_idx +  0]);
      (vtemp_result[1] - vzeroPointA * vadj[1]).store_unaligned(&output[col_idx +  4]);
      (vtemp_result[2] - vzeroPointA * vadj[2]).store_unaligned(&output[col_idx +  8]);
      (vtemp_result[3] - vzeroPointA * vadj[3]).store_unaligned(&output[col_idx + 12]);
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
#if 0
// Modified function
void GemmMatMulM1(const uint8_t *inputMatrixA, const int8_t *inputMatrixB,
                  size_t rowsA, size_t width, size_t colsB, uint8_t zeroPointA,
                  const uint8_t *zeroPointB, const float *b_scale_data,
                  bool is_b_scale_per_column, float *output) {

  vuint8_t vzeroPointA(zeroPointA);

  for (size_t col_idx = 0; col_idx < colsB; col_idx += 16) {
      vint32_t vtemp_result[4] = {};

      for (size_t k = 0; k < width; k += 4) {
        vuint8_t va_data = xsimd::bitwise_cast<uint8_t>(
            vuint32_t(*(uint32_t *)(inputMatrixA + k)));
        constexpr xsimd::batch_constant<uint8_t, arch, 0, 1, 4, 5, 8, 9, 12, 13,
                                        2, 3, 6, 7, 10, 11, 14, 15>
            mask;
        vuint8_t va_datap = xsimd::swizzle(va_data, mask);
        vuint8_t va_data_adj_lo = zip_lo(va_datap, vuint8_t(0));
        vuint8_t va_data_adj_hi = zip_hi(va_datap, vuint8_t(0));

        vint8_t vb_data0 = vint8_t::load_unaligned(
            &inputMatrixB[(k + 0) * colsB + col_idx + 0]);
        vint8_t vb_data1 = vint8_t::load_unaligned(
            &inputMatrixB[(k + 1) * colsB + col_idx + 0]);
        vint8_t vb_data2 = vint8_t::load_unaligned(
            &inputMatrixB[(k + 2) * colsB + col_idx + 0]);
        vint8_t vb_data3 = vint8_t::load_unaligned(
            &inputMatrixB[(k + 3) * colsB + col_idx + 0]);

        vint16_t vb_data_lo0 =
            xsimd::bit_cast<vint16_t>(zip_lo(vb_data0, vb_data1));
        vint16_t vb_data_lo1 =
            xsimd::bit_cast<vint16_t>(zip_lo(vb_data2, vb_data3));

        vint8_t vb_data0p =
            xsimd::bit_cast<vint8_t>(zip_lo(vb_data_lo0, vb_data_lo1));
        vint8_t vb_data1p =
            xsimd::bit_cast<vint8_t>(zip_hi(vb_data_lo0, vb_data_lo1));

        vint8_t vb_data0pp = xsimd::swizzle(vb_data0p, mask);
        vint8_t vb_data1pp = xsimd::swizzle(vb_data1p, mask);

        vtemp_result[0] = gemmology::maddw(
            va_data_adj_lo, zip_lo(vb_data0pp, vint8_t(0)), vtemp_result[0]);
        vtemp_result[0] = gemmology::maddw(
            va_data_adj_hi, zip_hi(vb_data0pp, vint8_t(0)), vtemp_result[0]);

        vtemp_result[1] = gemmology::maddw(
            va_data_adj_lo, zip_lo(vb_data1pp, vint8_t(0)), vtemp_result[1]);
        vtemp_result[1] = gemmology::maddw(
            va_data_adj_hi, zip_hi(vb_data1pp, vint8_t(0)), vtemp_result[1]);

        vtemp_result[0] = gemmology::maddw(
            vzeroPointA, zip_lo(vb_data0pp, vint8_t(0)), -vtemp_result[0]);
        vtemp_result[0] = -gemmology::maddw(
            vzeroPointA, zip_hi(vb_data0pp, vint8_t(0)), vtemp_result[0]);

        vtemp_result[1] = gemmology::maddw(
            vzeroPointA, zip_lo(vb_data1pp, vint8_t(0)), -vtemp_result[1]);
        vtemp_result[1] = -gemmology::maddw(
            vzeroPointA, zip_hi(vb_data1pp, vint8_t(0)), vtemp_result[1]);

        vint16_t vb_data_hi0 =
            xsimd::bit_cast<vint16_t>(zip_hi(vb_data0, vb_data1));
        vint16_t vb_data_hi1 =
            xsimd::bit_cast<vint16_t>(zip_hi(vb_data2, vb_data3));

        vint8_t vb_data2p =
            xsimd::bit_cast<vint8_t>(zip_lo(vb_data_hi0, vb_data_hi1));
        vint8_t vb_data3p =
            xsimd::bit_cast<vint8_t>(zip_hi(vb_data_hi0, vb_data_hi1));

        auto vb_data2pp = xsimd::swizzle(vb_data2p, mask);
        auto vb_data3pp = xsimd::swizzle(vb_data3p, mask);

        vtemp_result[2] = gemmology::maddw(
            va_data_adj_lo, zip_lo(vb_data2pp, vint8_t(0)), vtemp_result[2]);
        vtemp_result[2] = gemmology::maddw(
            va_data_adj_hi, zip_hi(vb_data2pp, vint8_t(0)), vtemp_result[2]);

        vtemp_result[3] = gemmology::maddw(
            va_data_adj_lo, zip_lo(vb_data3pp, vint8_t(0)), vtemp_result[3]);
        vtemp_result[3] = gemmology::maddw(
            va_data_adj_hi, zip_hi(vb_data3pp, vint8_t(0)), vtemp_result[3]);

        vtemp_result[2] = gemmology::maddw(
            vzeroPointA, zip_lo(vb_data2pp, vint8_t(0)), -vtemp_result[2]);
        vtemp_result[2] = -gemmology::maddw(
            vzeroPointA, zip_hi(vb_data2pp, vint8_t(0)), vtemp_result[2]);

        vtemp_result[3] = gemmology::maddw(
            vzeroPointA, zip_lo(vb_data3pp, vint8_t(0)), -vtemp_result[3]);
        vtemp_result[3] = -gemmology::maddw(
            vzeroPointA, zip_hi(vb_data3pp, vint8_t(0)), vtemp_result[3]);
      }

      vtemp_result[0].store_unaligned(&output[col_idx + 0]);
      vtemp_result[1].store_unaligned(&output[col_idx + 4]);
      vtemp_result[2].store_unaligned(&output[col_idx + 8]);
      vtemp_result[3].store_unaligned(&output[col_idx + 12]);
  }


  int32_t a_acc = 0;
  for (size_t k = 0; k < width; k += 1) {
    a_acc += inputMatrixA[k];
  }
  for (size_t col_idx = 0; col_idx < colsB; col_idx += 1) {
    output[col_idx] += -a_acc * zeroPointB[col_idx];
  }

  for (size_t col_idx = 0; col_idx < colsB; col_idx += 1) {
    output[col_idx] += zeroPointA * zeroPointB[col_idx] * width;
  }

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
#endif 

void displayMatrix(const int8_t *matrix, size_t rows, size_t cols) {
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      std::cout << static_cast<int>(matrix[row * cols + col]) << " ";
    }
    std::cout << std::endl;
  }
}

void displayMatrix(const uint8_t *matrix, size_t rows, size_t cols) {
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      std::cout << static_cast<int>(matrix[row * cols + col]) << " ";
    }
    std::cout << std::endl;
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

  if (rowsA == 1)
    return GemmMatMulM1(inputMatrixA, inputMatrixB, rowsA, width, colsB,
                        zeroPointA, zeroPointB, b_scale_data,
                        is_b_scale_per_column, output);

  // Transpose B to make further computation faster
  int8_t *b_transposed = new int8_t[colsB * width];
  for (size_t k = 0; k < width; k += vint8_t::size) {
    for (size_t n = 0; n < colsB; n += vint8_t::size) {
      vint8_t b_value[vint8_t::size];
      for (size_t vk = 0; vk < vint8_t::size; ++vk)
        b_value[vk] =
            vint8_t::load_unaligned(&inputMatrixB[(vk + k) * colsB + n]);
      xsimd::transpose(std::begin(b_value), std::end(b_value));
      for (size_t vk = 0; vk < vint8_t::size; ++vk)
        b_value[vk].store_unaligned(&b_transposed[(vk + n) * width + k]);
    }
  }

  // Precompute sum(b[:,k]) * zeroPointA
  int32_t *b_acc = new int32_t[colsB];
  vuint8_t vzeroPointA = zeroPointA;
  for (size_t col_idx = 0; col_idx < colsB; col_idx += 4) {

    vint32_t vb_acc[4] = {};
    for (size_t k = 0; k < width; k += vint8_t::size) {
      vb_acc[0] = gemmology::maddw(
          vzeroPointA,
          vint8_t::load_unaligned(&b_transposed[col_idx * width + k]),
          vb_acc[0]);
      vb_acc[1] = gemmology::maddw(
          vzeroPointA,
          vint8_t::load_unaligned(&b_transposed[(col_idx + 1) * width + k]),
          vb_acc[1]);
      vb_acc[2] = gemmology::maddw(
          vzeroPointA,
          vint8_t::load_unaligned(&b_transposed[(col_idx + 2) * width + k]),
          vb_acc[2]);
      vb_acc[3] = gemmology::maddw(
          vzeroPointA,
          vint8_t::load_unaligned(&b_transposed[(col_idx + 3) * width + k]),
          vb_acc[3]);
    }
    xsimd::transpose(std::begin(vb_acc), std::end(vb_acc));
    (vb_acc[0] + vb_acc[1] + vb_acc[2] + vb_acc[3])
        .store_unaligned(&b_acc[col_idx]);
  }

  for (size_t row_idx = 0; row_idx < rowsA; ++row_idx) {

    const uint8_t *a_row = inputMatrixA + row_idx * width;

    // Precompute a[k] {
    vint32_t va_acc = 0;
    for (size_t k = 0; k < width; k += vint8_t::size) {
      va_acc = gemmology::maddw(vuint8_t::load_unaligned(&a_row[k]),
                                vint8_t(-1), va_acc);
    }
    int32_t a_acc = reduce_add(va_acc);
    // }

    for (size_t col_idx = 0; col_idx < colsB; col_idx += 4) {
        const int8_t *b_col0 = b_transposed + (col_idx + 0) * width;
        const int8_t *b_col1 = b_transposed + (col_idx + 1) * width;
        const int8_t *b_col2 = b_transposed + (col_idx + 2) * width;
        const int8_t *b_col3 = b_transposed + (col_idx + 3) * width;
        vint32_t vtmp[4] = {}; // ???

        for (size_t k = 0; k < width; k += vint8_t::size) {
          vuint8_t a_value = vuint8_t::load_unaligned(&a_row[k]);
          vint8_t b_value0 = vint8_t::load_unaligned(&b_col0[k]);
          vint8_t b_value1 = vint8_t::load_unaligned(&b_col1[k]);
          vint8_t b_value2 = vint8_t::load_unaligned(&b_col2[k]);
          vint8_t b_value3 = vint8_t::load_unaligned(&b_col3[k]);
          vtmp[0] = gemmology::maddw(a_value, b_value0, vtmp[0]);
          vtmp[1] = gemmology::maddw(a_value, b_value1, vtmp[1]);
          vtmp[2] = gemmology::maddw(a_value, b_value2, vtmp[2]);
          vtmp[3] = gemmology::maddw(a_value, b_value3, vtmp[3]);
        }

        xsimd::transpose(std::begin(vtmp), std::end(vtmp));
        vint32_t vout = vtmp[0] + vtmp[1] + vtmp[2] + vtmp[3];
        if (is_b_scale_per_column) {

          vout +=
              (int32_t(zeroPointA) * width + a_acc) *
                  vint32_t{zeroPointB[col_idx + 0], zeroPointB[col_idx + 1],
                           zeroPointB[col_idx + 2], zeroPointB[col_idx + 3]} -
              vint32_t::load_unaligned(&b_acc[col_idx]);
        } else {
          vout += (int32_t(zeroPointA) * width + a_acc) *
                      vint32_t{zeroPointB[0], zeroPointB[0], zeroPointB[0],
                               zeroPointB[0]} -
                  vint32_t::load_unaligned(&b_acc[col_idx]);
        }
        vout.store_unaligned(&output[row_idx * colsB + col_idx]);
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

  delete[] b_transposed;
  delete[] b_acc;
}

void CompareMatMul(size_t rowsA, size_t width, size_t colsB, uint8_t zeroPointA,
                   bool profile) {
  if (width % 16 != 0) {
    throw std::runtime_error(
        "Width must be divisible by 16 for SIMD operations.");
  }

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
  for (int x = 0; x < (profile ? 100 : 1); ++x) {
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
    CompareMatMul(1, 1024, 1024, 123, profile);
  } else {
    CompareMatMul(1, 128, 256, 0, profile);
    CompareMatMul(1, 128, 256, 1, profile);
    // CompareMatMul(256, 256, 256, 0, profile);
    CompareMatMul(1, 1024, 1024, 0, profile);
    CompareMatMul(1, 1024, 1024, 123, profile);
    CompareMatMul(1, 1024, 4096, 123, profile);
  }

  return 0;
}
