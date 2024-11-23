#ifndef MATRIX_MULTIPLY_H
#define MATRIX_MULTIPLY_H

#include <cuda_runtime.h>
#include "cyber/cyber.h"

// 矩阵乘法函数声明：只需要矩阵大小 N 和 CUDA 流
void matrixMultiplyTest(int N, cudaStream_t stream = nullptr, std::string name_component = "");

#endif // MATRIX_MULTIPLY_H
