#include <iostream>
#include <fstream>
#include <memory>
#include <cstdlib>    // for rand()
#include <ctime>
#include <cuda_runtime.h>

// 以下是你的项目或 Apollo 框架头文件，根据需要修改
#include "cyber/case_study/test.h"
#include "cyber/time/time.h"
#include "cyber/croutine/croutine.h"
#include "cyber/common/log.h"
#include "libsmctrl.h"

// 声明多流的数量
#define STREAM_NUM 1
#define BLOCK_SIZE 16

//---------------------------------------------------------------------
// 以下示例依旧使用 initMatrix(), count_unique() 等函数
//---------------------------------------------------------------------
static void initMatrix(float *matrix, int M, int N){
    for(int i=0; i<M*N; i++){
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

static int sort_asc(const void* a, const void* b) {
  return *(uint8_t*)a - *(uint8_t*)b;
}

static int count_unique(uint8_t* arr, int len, int *uni_smids) {
  qsort(arr, len, 1, sort_asc);
  int num_uniq = 1;
  if (len != 0){
    uni_smids[0] = static_cast<int>(arr[0]);
  }
  for (int i = 0; i < len - 1; i++){
    num_uniq += (arr[i] != arr[i + 1]);
    if(arr[i] != arr[i+1]){
        uni_smids[num_uniq-1] = arr[i+1];
    }
  }
  return num_uniq;
}

//---------------------------------------------------------------------
// 这里是 test_component 类的实现
//---------------------------------------------------------------------
bool test_component::Init() {
  return true;
}

bool test_component::Proc() {

  // 为了演示，我们先确定矩阵大小
  const int M = 1024, N = 2048, K = 1024;
  float alpha = 1.0f;
  float beta  = 0.0f;

  // 准备多流（这里先演示 1 个流，你也可以改成 STREAM_NUM>1）
  cudaStream_t streams[STREAM_NUM];
  for(int i=0; i<STREAM_NUM; i++){
    cudaStreamCreate(&streams[i]);
  }
  // 如果想把 Apollo Croutine 当前流作为 streams[0]：
  // streams[0] = apollo::cyber::croutine::CRoutine::GetCurrentStream();

  // 分配并初始化 Host 内存
  float *h_A[STREAM_NUM], *h_B[STREAM_NUM], *h_C[STREAM_NUM];
  for(int i=0; i<STREAM_NUM; i++){
      h_A[i] = new float[M*K];
      h_B[i] = new float[K*N];
      h_C[i] = new float[M*N];
      initMatrix(h_A[i], M, K);
      initMatrix(h_B[i], K, N);
      initMatrix(h_C[i], M, N);
  }

  // 分配 Device 内存
  float *d_A[STREAM_NUM], *d_B[STREAM_NUM], *d_C[STREAM_NUM];
  for(int i=0; i<STREAM_NUM; i++){
    cudaMalloc(&d_A[i], M*K*sizeof(float));
    cudaMalloc(&d_B[i], K*N*sizeof(float));
    cudaMalloc(&d_C[i], M*N*sizeof(float));
  }

  // 下面分配记录 SM ID 的数组
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocksGrid((N+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE);
  size_t smids_size = blocksGrid.x * blocksGrid.y * sizeof(uint8_t);

  uint8_t *smids_d[STREAM_NUM];
  uint8_t *smids_h[STREAM_NUM];
  for(int i=0; i<STREAM_NUM; i++){
    cudaMalloc(&smids_d[i], smids_size);
    smids_h[i] = new uint8_t[smids_size];
  }

  // 打开一个文件，用来记录测试数据
//   std::ofstream ofs("/tmp/kernel_time.txt");
//   if (!ofs.is_open()) {
//     AERROR << "Failed to open kernel_time.txt for writing.";
//     return false;
//   }

  // 这里假设 GPU 总共有 20 个 SM，最大掩码就是 2^20 - 1 = 1048575
  // 你可以根据实际 GPU SM 数调整 total_sm
  const int total_sm = 40;

  // 在这里演示循环：从只启用 1 个 SM 到启用 total_sm 个 SM
  for (int sm_count = 1; sm_count <= total_sm; sm_count++) {
    // 掩码 mask：低 sm_count 位为 1
    //   sm_count=1 => mask=0b1
    //   sm_count=2 => mask=0b11
    //   ...
    //   sm_count=20 => mask=0xFFFFF (对于 20 个 SM)
    uint64_t mask = (1ULL << sm_count) - 1ULL;

    // 先把 Host 数据异步拷贝到 Device
    for(int i=0; i<STREAM_NUM; i++){
      cudaMemcpyAsync(d_A[i], h_A[i], M*K*sizeof(float), cudaMemcpyHostToDevice, streams[i]);
      cudaMemcpyAsync(d_B[i], h_B[i], K*N*sizeof(float), cudaMemcpyHostToDevice, streams[i]);
      cudaMemcpyAsync(d_C[i], h_C[i], M*N*sizeof(float), cudaMemcpyHostToDevice, streams[i]);
    }

    // 创建 CUDA event 记录时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 设置 GPU SM 掩码（libsmctrl_set_global_mask是反着来的）
    libsmctrl_set_global_mask(~mask);

    // 启动核函数
    cudaEventRecord(start, streams[0]);
    for(int i=0; i<STREAM_NUM; i++){
      gemm_multi_streams(blocksGrid, threadsPerBlock, streams[i],
                         d_A[i], d_B[i], d_C[i],
                         alpha, beta, M, N, K,
                         smids_d[i]);
    }
    cudaEventRecord(stop, streams[0]);

    // 同步等待所有流完成
    for(int i=0; i<STREAM_NUM; i++){
      cudaStreamSynchronize(streams[i]);
    }

    // 计算内核执行时间 (ms)
    float kernelExeTime = 0.0f;
    cudaEventElapsedTime(&kernelExeTime, start, stop);

    // 把结果写到文件 (例如 "SM_used=3    Kernel_time=123.45 ms")
    AINFO << "SM_used=" << sm_count 
        << "\tKernel_time=" << kernelExeTime << " ms\n";

    // 释放 event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 如果想要在下一轮继续复用同一批内存，只需重新拷贝或修改数据即可
    // 如果每次都要重新分配，也可以在这里释放，但通常不必
  }

  // 循环测试结束，关闭文件
//   ofs.close();

  // 现在拷回结果 (如有需要)
  for(int i=0; i<STREAM_NUM; i++){
    cudaMemcpyAsync(h_C[i], d_C[i], M*N*sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    cudaMemcpyAsync(smids_h[i], smids_d[i], smids_size, cudaMemcpyDeviceToHost, streams[i]);
  }

  // 同步，以确保拷回完成
  for(int i=0; i<STREAM_NUM; i++){
    cudaStreamSynchronize(streams[i]);
  }

  // 统计并打印 SM ID 分布（如有需要）
//   for(int i=0; i<STREAM_NUM; i++){
//     int *uniq_ids = new int[blocksGrid.x * blocksGrid.y];
//     int uniq_counts = count_unique(smids_h[i], blocksGrid.x * blocksGrid.y, uniq_ids);
//     std::cout << "---------------  Stream " << i << "-------------------" << std::endl;
//     std::cout << "The number of different SMs used: " << uniq_counts << std::endl;
//     std::cout << "Used SM IDs: ";
//     for(int j=0; j<uniq_counts; j++){
//       std::cout << uniq_ids[j] << " ";
//     }
//     std::cout << std::endl;
//     delete [] uniq_ids;
//   }

  // 释放内存：注意要正确区分 delete[] (Host) 和 cudaFree(Device)
  for(int i=0; i<STREAM_NUM; i++){
    // 先释放 Host
    delete [] h_A[i];
    delete [] h_B[i];
    delete [] h_C[i];
    delete [] smids_h[i];

    // 再释放 Device
    cudaFree(d_A[i]);
    cudaFree(d_B[i]);
    cudaFree(d_C[i]);
    cudaFree(smids_d[i]);

    // 销毁流
    cudaStreamDestroy(streams[i]);
  }


  // 这里可以把处理耗时写到 Apollo 日志或者 out_msg
  AINFO << "finish the test!";

  return true;
}
