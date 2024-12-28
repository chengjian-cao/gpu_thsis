#include "cyber/case_study/v2.h"
#include "libsmctrl.h"

#define STREAM_NUM 1
#define BLOCK_SIZE 16

bool v2_component::Init() {
  AINFO << "v2_component init";
  driver_writer_ = node_->CreateWriter<Driver>("/apollo/v2");
  return true;
}

void initMatrix(float *matrix, int M, int N){
    for(int i=0; i<M*N; i++){
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void releaseHostMem(float *A, float *B, float *C){
    delete [] A;
    delete [] B;
    delete [] C;
}

void allocDeviceMem(float *A, float *B, float *C, int M, int N, int K){
    cudaMalloc(&A, M*K*sizeof(float));
    cudaMalloc(&B, K*N*sizeof(float));
    cudaMalloc(&C, M*N*sizeof(float));
}

void releaseDeviceMem(float *A, float *B, float *C){
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

static int sort_asc(const void* a, const void* b) {
  return *(uint8_t*)a - *(uint8_t*)b;
}

static int count_unique(uint8_t* arr, int len, int *uni_smids) {
  // 1为sizeof(uint8)的1 byte;
  qsort(arr, len, 1, sort_asc);
  int num_uniq = 1;
  if(len != 0){
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

 

bool v2_component::Proc(const std::shared_ptr<Driver>& msg0) {
  auto out_msg = std::make_shared<Driver>();
  out_msg->set_msg_id(msg0->msg_id());
  apollo::cyber::Time now_time = Clock::Now();
  double start_time = now_time.ToMillisecond();

  // 打开输出文件。你也可以把它放到全局或 Init() 中，具体看需求。
  std::ofstream ofs("kernel_time.txt");
  if (!ofs.is_open()) {
    AERROR << "Failed to open output file: kernel_time.txt";
    return false;
  }

  // 如果你要测试从 “只使用 1 个 SM” 到 “使用所有 SM” 的不同效果，
  // 需要知道你的 GPU 一共有多少个 SM（比如 20 个）。
  // 这里假设你的 GPU 有 20 个 SM，因此最大掩码就是 2^20 - 1 = 1048575 (十进制)。
  // 当然你也可以把 20 换成其他数字，或者动态获取 GPU SM 个数。
  const int total_tpc = 40;  


  int M = 1024, N = 8192, K = 1024;

  float alpha = 1.0f;
  float beta = 0.0f;

  // record kernel execution time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate host memory and initialize data
  float *h_A[STREAM_NUM], *h_B[STREAM_NUM], *h_C[STREAM_NUM];
  for(int i=0; i<STREAM_NUM; i++){
      h_A[i] = new float[M*K];
      h_B[i] = new float[K*N];
      h_C[i] = new float[M*N];
      // allocHostMem(h_A[i], h_B[i], h_C[i], M, N, K);
      initMatrix(h_A[i], M, K);
      initMatrix(h_B[i], K, N);
      initMatrix(h_C[i], M, N);
  }

  // create streams
  cudaStream_t streams[STREAM_NUM];
  for(int i=0; i<STREAM_NUM; i++){
      cudaStreamCreate(&streams[i]);
  }

  // allocate Device memory and copy data
  float *d_A[STREAM_NUM], *d_B[STREAM_NUM], *d_C[STREAM_NUM];
  for(int i=0; i<STREAM_NUM; i++){
      // allocDeviceMem(d_A[i], d_B[i], d_C[i], M, N, K);
      cudaMalloc(&d_A[i], M*K*sizeof(float));
      cudaMalloc(&d_B[i], K*N*sizeof(float));
      cudaMalloc(&d_C[i], M*N*sizeof(float));
      cudaMemcpyAsync(d_A[i], h_A[i], M*K*sizeof(float), cudaMemcpyHostToDevice, streams[i]);
      cudaMemcpyAsync(d_B[i], h_B[i], K*N*sizeof(float), cudaMemcpyHostToDevice, streams[i]);
      cudaMemcpyAsync(d_C[i], h_C[i], M*N*sizeof(float), cudaMemcpyHostToDevice, streams[i]);
  }

     // grid blocks
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksGrid((N+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE);
    int smids_size = blocksGrid.x * blocksGrid.y * sizeof(uint8_t);


    // reocrding SM ID array
    uint8_t *smids_d[STREAM_NUM];
    for(int i=0; i<STREAM_NUM; i++){
        cudaMalloc(&smids_d[i], smids_size);
    }
    
    uint8_t *smids_h[STREAM_NUM];
    for(int i=0; i<STREAM_NUM; i++){
        smids_h[i] = new uint8_t[smids_size];
    }
  
    uint64_t mask_values[STREAM_NUM] = {1048575};
    // 1048575 TCP 0-19 SM 0-39
    // 1099511627775 - 1048575 TCP 20-39 SM 40-79

    cudaEventRecord(start, streams[0]);
    now_time = Clock::Now();
    double start_kernel = now_time.ToMillisecond();
    // 启动核函数
    for(int i=0; i<STREAM_NUM; i++){
        libsmctrl_set_global_mask(~(mask_values[i]));
        gemm_multi_streams(blocksGrid, threadsPerBlock, streams[i], d_A[i], d_B[i], d_C[i], alpha, beta, M, N, K, smids_d[i]);
    }

    // 获取当前时间作为核函数执行后的时间


    cudaEventRecord(stop, streams[0]);
    cudaEventSynchronize(stop);
    now_time = Clock::Now();
    double end_kernel = now_time.ToMillisecond();


    // 打印核函数结束后的时间和执行时间
    // if(!name_component.empty())

    float kernelExeTime = 0.0;
    cudaEventElapsedTime(&kernelExeTime, start, stop);
    AINFO << "v2" << " started kernel at " << (long long)start_kernel << " ms," << " ended kernel at " << (long long)end_kernel << " ms.";
    AINFO << "Kernel Execution time: " << kernelExeTime << "ms" ;

    // copy data from device to host
    for(int i=0; i<STREAM_NUM; i++){
        cudaMemcpyAsync(h_C[i], d_C[i], M*N*sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
        cudaMemcpyAsync(smids_h[i], smids_d[i], smids_size, cudaMemcpyDeviceToHost, streams[i]);
    }

    for(int i=0; i<STREAM_NUM; i++){
        int *uniq_ids = new int[smids_size];
        std::cout << "---------------  Stream " << i << "-------------------" << std::endl;
        int uniq_counts = count_unique(smids_h[i], smids_size, uniq_ids);
        std::cout << "The number of different SMs used: " << uniq_counts << std::endl;
        std::cout << "Used SM IDs: ";
        for(int i=0; i<uniq_counts; i++){
            std::cout << uniq_ids[i] << " ";
        }
        std::cout << std::endl;
        delete [] uniq_ids;
    }
    
    // release memory
    for(int i=0; i<STREAM_NUM; i++) {
        // 1) 释放 Host 上的指针
        delete[] h_A[i];
        delete[] h_B[i];
        delete[] h_C[i];
        delete[] smids_h[i];   // smids_h 也是在 host 上用 new[] 分配的

        // 2) 释放 Device 上的指针
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
        cudaFree(smids_d[i]);  // smids_d 也是用 cudaMalloc 分配在 device 上
    }


    now_time = Clock::Now();
    double end_time = now_time.ToMillisecond();

//   AINFO << "v2" << " started at " << (long long)start_time << " ms," << " ended at " << (long long)end_time << " ms."<< " The processing time is "<<end_time-start_time<<" ms.";
    out_msg->set_timestamp(msg0->start_time() + end_time - start_time );
    out_msg->set_start_time(msg0->start_time());
    driver_writer_->Write(out_msg);
    return true;
}