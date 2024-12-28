#include "gemm_multi_streams.h"
#include "libsmctrl.h"

// 一个简单的示例核函数：仅启动 20 个 block
// 每个 block 做一些无意义的循环计算，耗时取决于 blockDim/threadIdx
__global__ void limited_blocks_kernel(int iterations) {
    // 全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    // 做一大堆浮点运算，让该 block 计算时间变长
    for(int i = 0; i < iterations; i++){
        sum += sinf(idx * 0.001f + i);
    }
    // 防止编译器优化掉
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        // 仅仅打印(或者写到某个全局内存里)
    }
}

void limited_blocks_kernel(dim3 blocksGrid, dim3 threadsPerBlock, int iterations){
    limited_blocks_kernel<<<blocksGrid, threadsPerBlock>>>(iterations);
    return;
}

