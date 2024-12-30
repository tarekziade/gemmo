#include "gemmology/gemmology.h"
#include "xsimd/xsimd.hpp"
#include <cassert>
#include <chrono> // For timing
#include <cstdint>
#include <iomanip> // For formatted output
#include <iostream>
#include <vector>

using arch = xsimd::neon; //*/ xsimd::sse4_2;
using vuint8_t = xsimd::batch<uint8_t, arch>;
using vint8_t = xsimd::batch<int8_t, arch>;
using vint32_t = xsimd::batch<int32_t, arch>;


/** 
 * Naive implementation
 */
void MatMulFull(const uint8_t *inputMatrixA, const int8_t *inputMatrixB,
                size_t rowsA, size_t width, size_t colsB, uint8_t zeroPointA,
                const uint8_t *zeroPointB, const float *b_scale_data,
                bool is_b_scale_per_column, float *output) {

  float matrixScale = is_b_scale_per_column ? 0.0f : b_scale_data[0];
  int32_t matrixZeroPointB =
      is_b_scale_per_column ? 0 : static_cast<int32_t>(zeroPointB[0]);

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
          adjustedB -= matrixZeroPointB;
        }
        // Accumulate product
        tempResult += adjustedA * adjustedB;
      }

      float scaledResult = tempResult;
      if (is_b_scale_per_column) {
        scaledResult *= b_scale_data[colIndex];
      } else {
        scaledResult *= matrixScale;
      }

      // Store the scaled result in y_data
      output[rowIndex * colsB + colIndex] = scaledResult;
    }
  }
}

/**
 * Gemmology implementation
 */
void MatMulFullFloat(const uint8_t *inputMatrixA, const int8_t *inputMatrixB,
                     size_t rowsA, size_t width, size_t colsB,
                     uint8_t zeroPointA, const uint8_t *zeroPointB,
                     const float *b_scale_data, bool is_b_scale_per_column,
                     float *output) {
  vuint8_t vzeroPointA = zeroPointA;

  // Transpose B to make further computation faster
  int8_t *b_transposed = nullptr;
  // should assert
  posix_memalign(reinterpret_cast<void **>(&b_transposed), 64,
                 width * colsB * sizeof(int8_t));

  for (size_t k = 0; k < width; k += vint8_t::size) {
    for (size_t n = 0; n < colsB; n += vint8_t::size) {
      vint8_t b_value[vint8_t::size];

      // Load the input data into both sets of batches
      for (size_t vk = 0; vk < vint8_t::size; ++vk) {
        b_value[vk] =
            vint8_t::load_aligned(&inputMatrixB[(vk + k) * colsB + n]);
      }
      xsimd::transpose(std::begin(b_value), std::end(b_value));
      // Store the transposed values from transpose2
      for (size_t vk = 0; vk < vint8_t::size; ++vk) {
        b_value[vk].store_aligned(&b_transposed[(vk + n) * width + k]);
      }
    }
  }
  // Precompute sum(b[:,k]) * zeroPointA
  int32_t *b_acc = new int32_t[colsB];
  for (size_t col_idx = 0; col_idx < colsB; col_idx += 1) {
    vint32_t vb_acc = 0;
    for (size_t k = 0; k < width; k += vint8_t::size) {
      vb_acc = gemmology::maddw(
          vuint8_t(1),
          vint8_t::load_aligned(&b_transposed[col_idx * width + k]), vb_acc);
    }
    b_acc[col_idx] = reduce_add(vb_acc) * (int32_t)zeroPointA;
  }

  for (size_t row_idx = 0; row_idx < rowsA; ++row_idx) {
    const uint8_t *a_row = inputMatrixA + row_idx * width;

    for (size_t col_idx = 0; col_idx < colsB; col_idx += 4) {
      const int8_t *b_col0 = b_transposed + (col_idx + 0) * width;
      const int8_t *b_col1 = b_transposed + (col_idx + 1) * width;
      const int8_t *b_col2 = b_transposed + (col_idx + 2) * width;
      const int8_t *b_col3 = b_transposed + (col_idx + 3) * width;
      vint32_t vtmp0 = 0;
      vint32_t vtmp1 = 0;
      vint32_t vtmp2 = 0;
      vint32_t vtmp3 = 0;

      for (size_t k = 0; k < width; k += vint8_t::size) {
        vuint8_t a_value = vuint8_t::load_unaligned(&a_row[k]);
        vint8_t b_value0 = vint8_t::load_aligned(&b_col0[k]);
        vint8_t b_value1 = vint8_t::load_aligned(&b_col1[k]);
        vint8_t b_value2 = vint8_t::load_aligned(&b_col2[k]);
        vint8_t b_value3 = vint8_t::load_aligned(&b_col3[k]);
        vtmp0 = gemmology::maddw(a_value, b_value0, vtmp0);
        vtmp1 = gemmology::maddw(a_value, b_value1, vtmp1);
        vtmp2 = gemmology::maddw(a_value, b_value2, vtmp2);
        vtmp3 = gemmology::maddw(a_value, b_value3, vtmp3);
      }

      output[row_idx * colsB + col_idx + 0] = reduce_add(vtmp0);
      output[row_idx * colsB + col_idx + 1] = reduce_add(vtmp1);
      output[row_idx * colsB + col_idx + 2] = reduce_add(vtmp2);
      output[row_idx * colsB + col_idx + 3] = reduce_add(vtmp3);
    }

    // Precompute a[k] {
    vint32_t va_acc = 0;
    for (size_t k = 0; k < width; k += vint8_t::size) {
      va_acc = gemmology::maddw(vuint8_t::load_unaligned(&a_row[k]),
                                vint8_t(-1), va_acc);
    }
    int32_t a_acc = reduce_add(va_acc);
    // }

    // Apply offsets {
    for (size_t col_idx = 0; col_idx < colsB; col_idx += 1) {
      uint8_t b_offset =
          is_b_scale_per_column ? zeroPointB[col_idx] : zeroPointB[0];

      // Perform offset correction using float calculations
      float offset_correction =
          (static_cast<float>(zeroPointA) * width + static_cast<float>(a_acc)) *
              static_cast<float>(b_offset) -
          static_cast<float>(b_acc[col_idx]);

      float scale =
          is_b_scale_per_column ? b_scale_data[col_idx] : b_scale_data[0];
      output[row_idx * colsB + col_idx] += offset_correction;
      output[row_idx * colsB + col_idx] *= scale;
    }
    // }
  }

  delete[] b_transposed;
  delete[] b_acc;
}


void CompareMatMul(size_t rowsA, size_t width, size_t colsB, uint8_t zeroPointA) {
  uint8_t *zeroPointB = new uint8_t[colsB];
  for (size_t i = 0; i < colsB; ++i) {
    zeroPointB[i] = 0;
  }
  const bool is_b_scale_per_column = false;

  uint8_t *inputMatrixA = new uint8_t[rowsA * width];
  // Matrix A: rowsA x width
  for (size_t row = 0; row < rowsA; ++row) {
    for (size_t col = 0; col < width; ++col) {
      inputMatrixA[row * width + col] = col + 1; // Fill row by row
    }
  }

  int8_t *inputMatrixB = new int8_t[width * colsB];
  // Matrix B: width x colsB
  for (size_t row = 0; row < width; ++row) {
    for (size_t col = 0; col < colsB; ++col) {
      inputMatrixB[row * colsB + col] =
          row * colsB + col + 1; // Fill row by row
    }
  }

  float *b_scale_data = new float[colsB];
  for (size_t i = 0; i < colsB; ++i) {
    b_scale_data[i] = 1.0f;
  }

  std::vector<float> output1(rowsA * colsB, 0);
  std::vector<float> output2(rowsA * colsB, 0);

  auto start = std::chrono::high_resolution_clock::now();
  // for (int x = 0; x < 1000; ++x) {
  MatMulFull(inputMatrixA, inputMatrixB, rowsA, width, colsB, zeroPointA,
             zeroPointB, b_scale_data, is_b_scale_per_column,

             output1.data());
  //}
  auto end = std::chrono::high_resolution_clock::now();
  auto duration1 =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  std::cout << "MatMulFull took " << duration1 << " microseconds." << std::endl;

  start = std::chrono::high_resolution_clock::now();
  // for (int x = 0; x < 1000; ++x) {
  MatMulFullFloat(inputMatrixA, inputMatrixB, rowsA, width, colsB, zeroPointA,
                  zeroPointB, b_scale_data, is_b_scale_per_column,

                  output2.data());
  //}
  end = std::chrono::high_resolution_clock::now();
  auto duration2 =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  std::cout << "MatMulFullFloat took " << duration2 << " microseconds."
            << std::endl;

  for (size_t i = 0; i < rowsA * colsB; ++i) {
    int8_t diff = output1[i] - output2[i];
    if (diff == 0) {
      continue;
    }
    std::cout << "Index " << i << ": MatMulFull=" << output1[i]
              << ", MatMulFullFloat=" << output2[i]
              << ", Difference=" << std::abs(output1[i] - output2[i])
              << std::endl;
    throw std::runtime_error("Different results!");
  }
  if (std::equal(output1.begin(), output1.end(), output2.begin(),
                 [](float a, float b) { return std::abs(a - b) < 1e-5; })) {
    std::cout << "Both implementations produce the same result!" << std::endl;
  } else {
    std::cout << "The implementations produce different results!" << std::endl;
  }

  free(inputMatrixA);
  free(inputMatrixB);
}

int main() {
  CompareMatMul(1, 64, 1024, 123);
  CompareMatMul(1, 1024, 1024, 123);
  return 0;
}

