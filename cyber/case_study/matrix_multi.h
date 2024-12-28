#ifndef MATRIX_MULTI_H
#define MATRIX_MULTI_H

#include <cuda_runtime.h>
#include "cyber/cyber.h"


void matrix_multi(int n, std::string name_component, uint64_t mask_1,cudaStream_t stream);

#endif // MATRIX_MULTI_H