#include "cyber/case_study/v4.h"
#include "libsmctrl.h"

#define STREAM_NUM 1
#define BLOCK_SIZE 16
bool v4_component::Init() {
  // AINFO << "v4_component init";
  driver_writer_ = node_->CreateWriter<Driver>("/apollo/v4");
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

bool v4_component::Proc(const std::shared_ptr<Driver>& msg0) {
    auto out_msg = std::make_shared<Driver>();
    out_msg->set_msg_id(msg0->msg_id());
    out_msg->set_start_time(msg0->start_time());
    apollo::cyber::Time now_time = Clock::Now();
    double start_time = now_time.ToMillisecond();
    int sizeA = 128; // 第一矩阵的行数和第二矩阵的列数
    int sizeB = 128; // 第一矩阵的列数和第二矩阵的行数
    // 执行矩阵生成与相乘
    Matrix result = MatrixOperations::generateAndMultiplyMatrices(sizeA, sizeB);

//   cudaStream_t stream = apollo::cyber::croutine::CRoutine::GetCurrentStream();
//    int M = 1024, N = 8192, K = 1024;

//   float alpha = 1.0f;
//   float beta = 0.0f;

//   // record kernel execution time
//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);

//   // allocate host memory and initialize data
//   float *h_A[STREAM_NUM], *h_B[STREAM_NUM], *h_C[STREAM_NUM];
//   for(int i=0; i<STREAM_NUM; i++){
//       h_A[i] = new float[M*K];
//       h_B[i] = new float[K*N];
//       h_C[i] = new float[M*N];
//       // allocHostMem(h_A[i], h_B[i], h_C[i], M, N, K);
//       initMatrix(h_A[i], M, K);
//       initMatrix(h_B[i], K, N);
//       initMatrix(h_C[i], M, N);
//   }

//     // create streams
//     cudaStream_t streams[STREAM_NUM];
//     for(int i=0; i<STREAM_NUM; i++){
//         cudaStreamCreate(&streams[i]);
//     }
//     // streams[0] = stream;
//   // allocate Device memory and copy data
//   float *d_A[STREAM_NUM], *d_B[STREAM_NUM], *d_C[STREAM_NUM];
//   for(int i=0; i<STREAM_NUM; i++){
//       // allocDeviceMem(d_A[i], d_B[i], d_C[i], M, N, K);
//       cudaMalloc(&d_A[i], M*K*sizeof(float));
//       cudaMalloc(&d_B[i], K*N*sizeof(float));
//       cudaMalloc(&d_C[i], M*N*sizeof(float));
//       cudaMemcpyAsync(d_A[i], h_A[i], M*K*sizeof(float), cudaMemcpyHostToDevice, streams[i]);
//       cudaMemcpyAsync(d_B[i], h_B[i], K*N*sizeof(float), cudaMemcpyHostToDevice, streams[i]);
//       cudaMemcpyAsync(d_C[i], h_C[i], M*N*sizeof(float), cudaMemcpyHostToDevice, streams[i]);
//   }



//      // grid blocks
//     dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
//     dim3 blocksGrid((N+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE);
//     int smids_size = blocksGrid.x * blocksGrid.y * sizeof(uint8_t);


//     // reocrding SM ID array
//     uint8_t *smids_d[STREAM_NUM];
//     for(int i=0; i<STREAM_NUM; i++){
//         cudaMalloc(&smids_d[i], smids_size);
//     }
    
//     uint8_t *smids_h[STREAM_NUM];
//     for(int i=0; i<STREAM_NUM; i++){
//         smids_h[i] = new uint8_t[smids_size];
//     }
  
//     uint64_t mask_values[STREAM_NUM] = {1099511627775 - 1048575};
//     // 1048575 TCP 0-19 SM 0-39
//     // 1099511627775 - 1048575 TCP 20-39 SM 40-79

//     cudaEventRecord(start, streams[0]);
//     now_time = Clock::Now();
//     double start_kernel = now_time.ToMillisecond();

//     // 启动核函数
//     for(int i=0; i<STREAM_NUM; i++){
//         libsmctrl_set_global_mask(~(mask_values[i]));
//         gemm_multi_streams(blocksGrid, threadsPerBlock, streams[i], d_A[i], d_B[i], d_C[i], alpha, beta, M, N, K, smids_d[i]);
//     }

//     // 获取当前时间作为核函数执行后的时间

//     cudaEventRecord(stop, streams[0]);
//     cudaEventSynchronize(stop);
//     now_time = Clock::Now();
//     double end_kernel = now_time.ToMillisecond();
 
//     // 打印核函数结束后的时间和执行时间
//     // if(!name_component.empty())

//     float kernelExeTime = 0.0;
//     cudaEventElapsedTime(&kernelExeTime, start, stop);
//     AINFO << "v4" << " started kernel at " << (long long)start_kernel << " ms," << " ended kernel at " << (long long)end_kernel << " ms.";
//     AINFO << "Kernel Execution time: " << kernelExeTime << "ms" ;
    

//     // copy data from device to host
//     for(int i=0; i<STREAM_NUM; i++){
//         cudaMemcpyAsync(h_C[i], d_C[i], M*N*sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
//         cudaMemcpyAsync(smids_h[i], smids_d[i], smids_size, cudaMemcpyDeviceToHost, streams[i]);
//     }

//     for(int i=0; i<STREAM_NUM; i++){
//         int *uniq_ids = new int[smids_size];
//         std::cout << "---------------  Stream " << i << "-------------------" << std::endl;
//         int uniq_counts = count_unique(smids_h[i], smids_size, uniq_ids);
//         std::cout << "The number of different SMs used: " << uniq_counts << std::endl;
//         std::cout << "Used SM IDs: ";
//         for(int i=0; i<uniq_counts; i++){
//             std::cout << uniq_ids[i] << " ";
//         }
//         std::cout << std::endl;
//         delete [] uniq_ids;
//     }
    
//     // release memory
//     for(int i=0; i<STREAM_NUM; i++){
//         delete [] h_A[i], h_B[i], h_C[i];
//         cudaFree(h_A[i]);
//         cudaFree(h_B[i]);
//         cudaFree(h_C[i]);
//         delete [] smids_h[i];
//         cudaFree(smids_d[i]);
//     }

    now_time = Clock::Now();
    double end_time = now_time.ToMillisecond();

//   AINFO << "v4" << " mem started at " << (long long)start_time << " ms," << " ended at " << (long long)end_time << " ms."<< " The processing time is "<<end_time-start_time<<" ms.";

    out_msg->set_timestamp(msg0->start_time() + end_time - start_time );
    driver_writer_->Write(out_msg);
    // AINFO << "v4_component: Read drivermsg->"
    //   << msg0->ShortDebugString();
    return true;
}