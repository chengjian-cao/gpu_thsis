#ifndef LIMITED_BLOCKS_KERNEL_H
#define LIMITED_BLOCKS_KERNEL_H

#include <cuda_runtime.h>
#include "cyber/cyber.h"


void limited_blocks_kernel(dim3 blocksGrid, dim3 threadsPerBlock, int iterations);

#endif // LIMITED_BLOCKS_KERNEL_H