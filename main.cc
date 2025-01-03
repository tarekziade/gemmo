#include "gemmology/gemmology.h"
#include "xsimd/xsimd.hpp"
#include <arm_neon.h>
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

int32x4_t safe_vusdotq_s32(uint8x16_t x, uint8x16_t y, int32x4_t z) {
  int16x8_t tl = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(x))),
                           vmovl_s8(vget_low_s8(y)));
  int16x8_t th = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(x))),
                           vmovl_s8(vget_high_s8(y)));
  return vpadalq_s16(vpadalq_s16(z, tl), th);
}

/**
 * Gemmology implementation
 */
void GemmMatMulM1(const uint8_t *inputMatrixA, const int8_t *inputMatrixB,
                size_t rowsA, size_t width, size_t colsB, uint8_t zeroPointA,
                const uint8_t *zeroPointB, const float *b_scale_data,
                bool is_b_scale_per_column, float *output) {
   vuint8_t vzeroPointA = zeroPointA;  // ???

  // Transpose B to make further computation faster
  int8_t *b_transposed = new int8_t[colsB * width];
  for (size_t k = 0; k < width; k += vint8_t::size) {
    for (size_t n = 0; n < colsB; n += vint8_t::size) {
      vint8_t b_value[vint8_t::size];
      for (size_t vk = 0; vk < vint8_t::size; ++vk)
        b_value[vk] = vint8_t::load_unaligned(&inputMatrixB[(vk + k) * colsB + n]);
      xsimd::transpose(std::begin(b_value), std::end(b_value));
      for (size_t vk = 0; vk < vint8_t::size; ++vk)
        b_value[vk].store_unaligned(&b_transposed[(vk + n) * width + k]);
    }
  }

  const uint8_t *a_row = inputMatrixA;

  // Precompute a[k] {
  vint32_t va_acc = 0;
  for (size_t k = 0; k < width; k += vint8_t::size) {
    va_acc = safe_vusdotq_s32(vuint8_t::load_unaligned(&a_row[k]), vint8_t(-1),
                              va_acc);
  }
  int32_t a_acc = int32_t(zeroPointA) * width + reduce_add(va_acc);
  // }

  for (size_t col_idx = 0; col_idx < colsB; col_idx += 4) {
    vint32_t vtmp[4] = {};
    const int8_t *b_col0 = b_transposed + (col_idx + 0) * width;
    const int8_t *b_col1 = b_transposed + (col_idx + 1) * width;
    const int8_t *b_col2 = b_transposed + (col_idx + 2) * width;
    const int8_t *b_col3 = b_transposed + (col_idx + 3) * width;

    for (size_t k = 0; k < width; k += vint8_t::size) {
      vuint8_t a_value = vuint8_t::load_unaligned(&a_row[k]);
      vint8_t b_value0 = vint8_t::load_unaligned(&b_col0[k]);
      vint8_t b_value1 = vint8_t::load_unaligned(&b_col1[k]);
      vint8_t b_value2 = vint8_t::load_unaligned(&b_col2[k]);
      vint8_t b_value3 = vint8_t::load_unaligned(&b_col3[k]);

      vtmp[0] = safe_vusdotq_s32(vuint8_t(zeroPointA), -b_value0, vtmp[0]);
      vtmp[1] = safe_vusdotq_s32(vuint8_t(zeroPointA), -b_value1, vtmp[1]);
      vtmp[2] = safe_vusdotq_s32(vuint8_t(zeroPointA), -b_value2, vtmp[2]);
      vtmp[3] = safe_vusdotq_s32(vuint8_t(zeroPointA), -b_value3, vtmp[3]);

      vtmp[0] = safe_vusdotq_s32(a_value, b_value0, vtmp[0]);
      vtmp[1] = safe_vusdotq_s32(a_value, b_value1, vtmp[1]);
      vtmp[2] = safe_vusdotq_s32(a_value, b_value2, vtmp[2]);
      vtmp[3] = safe_vusdotq_s32(a_value, b_value3, vtmp[3]);
    }

    xsimd::transpose(std::begin(vtmp), std::end(vtmp));
    vint32_t vout = vtmp[0] + vtmp[1] + vtmp[2] + vtmp[3];
    vout += a_acc * vint32_t{zeroPointB[col_idx +0], zeroPointB[col_idx +1], zeroPointB[col_idx +2], zeroPointB[col_idx +3]};
    vout.store_unaligned(&output[col_idx]);
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


}

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


  vuint8_t vzeroPointA = zeroPointA;  // ???

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
  for (size_t col_idx = 0; col_idx < colsB; col_idx += 4) {
    vint32_t vb_acc[4] = {};
    for (size_t k = 0; k < width; k += vint8_t::size) {
      vb_acc[0] = safe_vusdotq_s32(
          vuint8_t(1),
          vint8_t::load_unaligned(&b_transposed[(col_idx + 0) * width + k]),
          vb_acc[0]);
      vb_acc[1] = safe_vusdotq_s32(
          vuint8_t(1),
          vint8_t::load_unaligned(&b_transposed[(col_idx + 1) * width + k]),
          vb_acc[1]);
      vb_acc[2] = safe_vusdotq_s32(
          vuint8_t(1),
          vint8_t::load_unaligned(&b_transposed[(col_idx + 2) * width + k]),
          vb_acc[2]);
      vb_acc[3] = safe_vusdotq_s32(
          vuint8_t(1),
          vint8_t::load_unaligned(&b_transposed[(col_idx + 3) * width + k]),
          vb_acc[3]);
    }
    xsimd::transpose(std::begin(vb_acc), std::end(vb_acc));
    (static_cast<int32_t>(zeroPointA) *
     (vb_acc[0] + vb_acc[1] + vb_acc[2] + vb_acc[3]))
        .store_unaligned(&b_acc[col_idx]);
  }

  for (size_t row_idx = 0; row_idx < rowsA; ++row_idx) {
    const uint8_t *a_row = inputMatrixA + row_idx * width;

    // Precompute a[k] {
    vint32_t va_acc = 0;
    for (size_t k = 0; k < width; k += vint8_t::size) {
      va_acc = safe_vusdotq_s32(vuint8_t::load_unaligned(&a_row[k]),
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
        vtmp[0] = safe_vusdotq_s32(a_value, b_value0, vtmp[0]);
        vtmp[1] = safe_vusdotq_s32(a_value, b_value1, vtmp[1]);
        vtmp[2] = safe_vusdotq_s32(a_value, b_value2, vtmp[2]);
        vtmp[3] = safe_vusdotq_s32(a_value, b_value3, vtmp[3]);
      }

      xsimd::transpose(std::begin(vtmp), std::end(vtmp));
      vint32_t vout = vtmp[0] + vtmp[1] + vtmp[2] + vtmp[3];
      if (is_b_scale_per_column) {

        vout += (int32_t(zeroPointA) * width + a_acc) *
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
    zeroPointB[i] = 0;
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

  start = std::chrono::high_resolution_clock::now();
  for (int x = 0; x < (profile ? 100 : 1); ++x) {
    GemmMatMul(inputMatrixA, inputMatrixB, rowsA, width, colsB, zeroPointA,
               zeroPointB, b_scale_data, is_b_scale_per_column,

               output2.data());
  }
  end = std::chrono::high_resolution_clock::now();
  auto duration2 =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  std::cout << "GemmMatMul took " << duration2 << " microseconds." << std::endl;

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
    std::cout << "The implementations produce different results!" << std::endl;
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
    CompareMatMul(1, 256, 256, 0, profile);
    CompareMatMul(256, 256, 256, 0, profile);
    CompareMatMul(1, 1024, 1024, 0, profile);
    CompareMatMul(1, 1024, 1024, 123, profile);
    CompareMatMul(1, 1024, 4096, 123, profile);
  }

  return 0;
}
