#ifndef GEMM_MULTI_STREAMS_H
#define GEMM_MULTI_STREAMS_H

#include <cuda_runtime.h>
#include "cyber/cyber.h"


void gemm_multi_streams(dim3 blocksGrid, dim3 threadsPerBlock, cudaStream_t streams, float *d_A, float *d_B, float *d_C, float alpha, float beta, int M, int N, int K, uint8_t *smids_d);

#endif // GEMM_MULTI_STREAMS_H
