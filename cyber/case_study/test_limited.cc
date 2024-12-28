#include <iostream>
#include <fstream>
#include <memory>
#include <cstdlib>    // for rand()
#include <ctime>
#include <cuda_runtime.h>

// 以下是你的项目或 Apollo 框架头文件，根据需要修改
#include "cyber/case_study/test_limited.h"
#include "cyber/time/time.h"
#include "cyber/croutine/croutine.h"
#include "cyber/common/log.h"
#include "libsmctrl.h"



//---------------------------------------------------------------------
// 这里是 test_component 类的实现
//---------------------------------------------------------------------
bool test_limited_component::Init() {
  return true;
}

bool test_limited_component::Proc() {

    // 要写入的输出文件
    std::ofstream ofs("/tmp/limited_blocks_time.txt");
    if (!ofs.is_open()) {
        std::cerr << "Failed to open limited_blocks_time.txt" << std::endl;
        return 1;
    }

    // 以下设定：只启动 20 个 block，每个 block 256 个线程
    // 这样 GPU 最多只能同时运行 20 个 block
    int gridSize   = 20;
    int blockSize  = 256;
    int iterations = 300000; // 循环次数，越大 kernel 越耗时

    // CUDA events 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 假设 GPU 有 40 个 SM
    int total_sm = 40;

    for(int sm_count = 1; sm_count <= total_sm; sm_count++){
        // 准备 SM 掩码
        // 低 sm_count 位=1 => 启用 sm_count 个 SM
        uint64_t mask = (1ULL << sm_count) - 1ULL;
        // libsmctrl_set_global_mask通常是反着来：
        libsmctrl_set_global_mask(~mask);

        cudaEventRecord(start);
        // 启动 kernel
        limited_blocks_kernel(gridSize, blockSize,iterations);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        // 记录
        // AINFO << "SM_used=" << sm_count << ", kernel_time=" << ms << " ms\n";

        std::cout << "SM_used=" << sm_count 
                  << "  kernel_time=" << ms << " ms\n";
    }

    // 还原 SM：启用全部
    libsmctrl_set_global_mask(0ULL);

    // 释放event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    ofs.close();


  return true;
}