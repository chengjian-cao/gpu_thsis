#include "gemm_multi_streams.h"
#include "libsmctrl.h"

#define STREAM_NUM 2
#define BLOCK_SIZE 16

// CUDA 核函数：计算 C = A * B
__global__ void gemmKernel(float *A, float *B, float *C, float alpha, float beta, int M, int N, int K, uint8_t *smids){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M && col < N){
        //get sm id
        if(threadIdx.x==1 && threadIdx.y == 1){
            int smid;
            asm("mov.u32 %0, %%smid;" : "=r"(smid));
            smids[gridDim.x * blockIdx.y + blockIdx.x] = smid;
            // smids[0] = smid;
        }
        // thread execution
        float Cvalue = 0.0;
        for(int i=0; i<K; i++){
            Cvalue += A[row*K+i] * B[i*N + col];
        }
        C[row*N+col] = alpha * Cvalue + beta * C[row*N +col];
    }
}



// void initMatrix(float *matrix, int M, int N){
//     for(int i=0; i<M*N; i++){
//         matrix[i] = static_cast<float>(rand()) / RAND_MAX;
//     }
// }

// void releaseHostMem(float *A, float *B, float *C){
//     delete [] A;
//     delete [] B;
//     delete [] C;
// }

// void allocDeviceMem(float *A, float *B, float *C, int M, int N, int K){
//     cudaMalloc(&A, M*K*sizeof(float));
//     cudaMalloc(&B, K*N*sizeof(float));
//     cudaMalloc(&C, M*N*sizeof(float));
// }

// void releaseDeviceMem(float *A, float *B, float *C){
//     cudaFree(A);
//     cudaFree(B);
//     cudaFree(C);
// }

// static int sort_asc(const void* a, const void* b) {
//   return *(uint8_t*)a - *(uint8_t*)b;
// }


// static int count_unique(uint8_t* arr, int len, int *uni_smids) {
//   // 1为sizeof(uint8)的1 byte;
//   qsort(arr, len, 1, sort_asc);
//   int num_uniq = 1;
//   if(len != 0){
//     uni_smids[0] = static_cast<int>(arr[0]);
//   }
//   for (int i = 0; i < len - 1; i++){
//     num_uniq += (arr[i] != arr[i + 1]);
//     if(arr[i] != arr[i+1]){
//         uni_smids[num_uniq-1] = arr[i+1];
//     }
//   }
//   return num_uniq;
    
// }

void gemm_multi_streams(dim3 blocksGrid, dim3 threadsPerBlock, cudaStream_t stream, float *d_A, float *d_B, float *d_C, float alpha, float beta, int M, int N, int K, uint8_t *smids_d) {

//     int M, N, K;
//     N = n;
//     M = K = 1024;
//     float alpha = 1.0f;
//     float beta = 0.0f;

//     // record kernel execution time
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     // allocate host memory and initialize data
//     float *h_A[STREAM_NUM], *h_B[STREAM_NUM], *h_C[STREAM_NUM];
//     for(int i=0; i<STREAM_NUM; i++){
//         h_A[i] = new float[M*K];
//         h_B[i] = new float[K*N];
//         h_C[i] = new float[M*N];
//         // allocHostMem(h_A[i], h_B[i], h_C[i], M, N, K);
//         initMatrix(h_A[i], M, K);
//         initMatrix(h_B[i], K, N);
//         initMatrix(h_C[i], M, N);
//     }

//     // allocate Device memory and copy data
//     float *d_A[STREAM_NUM], *d_B[STREAM_NUM], *d_C[STREAM_NUM];
//     for(int i=0; i<STREAM_NUM; i++){
//         // allocDeviceMem(d_A[i], d_B[i], d_C[i], M, N, K);
//         cudaMalloc(&d_A[i], M*K*sizeof(float));
//         cudaMalloc(&d_B[i], K*N*sizeof(float));
//         cudaMalloc(&d_C[i], M*N*sizeof(float));
//         cudaMemcpy(d_A[i], h_A[i], M*K*sizeof(float), cudaMemcpyHostToDevice);
//         cudaMemcpy(d_B[i], h_B[i], K*N*sizeof(float), cudaMemcpyHostToDevice);
//         cudaMemcpy(d_C[i], h_C[i], M*N*sizeof(float), cudaMemcpyHostToDevice);
//     }

//     // create streams
//     cudaStream_t streams[STREAM_NUM];
//     for(int i=0; i<STREAM_NUM; i++){
//         cudaStreamCreate(&streams[i]);
//     }

//    // grid blocks
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

//     uint64_t mask_values[STREAM_NUM] = {1048575, 1099511627775 - 1048575};
//     //192, 768, 3072, 12288
//     // The number of different SMs used: 4
//     // Used SM IDs: 12 13 14 15 
//     // ---------------  Stream 1-------------------
//     // The number of different SMs used: 4
//     // Used SM IDs: 16 17 18 19 
//     // ---------------  Stream 2-------------------
//     // The number of different SMs used: 4
//     // Used SM IDs: 20 21 22 23 
//     // ---------------  Stream 3-------------------
//     // The number of different SMs used: 4
//     // Used SM IDs: 24 25 26 27 

//     // 获取当前时间作为核函数启动前的时间
//     apollo::cyber::Time now_time = apollo::cyber::Time::Now();
//     double start_kernel = now_time.ToMillisecond();
//     cudaEventRecord(start);

//     // 启动核函数
//     for(int i=0; i<STREAM_NUM; i++){
//         libsmctrl_set_global_mask(~(mask_values[i]));
//         gemmKernel<<<blocksGrid, threadsPerBlock, 0, streams[i]>>>(d_A[i], d_B[i], d_C[i], alpha, beta, M, N, K, smids_d[i]);
//     }

//     // 获取当前时间作为核函数执行后的时间
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     apollo::cyber::Time end_time = apollo::cyber::Time::Now();
//     double end_kernel = end_time.ToMillisecond();

//     // 打印核函数结束后的时间和执行时间
//     // if(!name_component.empty())
//     AINFO << name_component << " kernel launch started at " << (long long)start_kernel << " ms," << " ended at " << (long long)end_kernel << " ms."<< " The processing time is "<<end_kernel-start_kernel<<" ms.";
//     // float kernelExeTime = 0.0;
//     // cudaEventElapsedTime(&kernelExeTime, start, stop);
//     // AINFO << "Kernel Execution time: " << kernelExeTime << "ms" ;

//     // copy data from device to host
//     for(int i=0; i<STREAM_NUM; i++){
//         cudaMemcpy(h_C[i], d_C[i], M*N*sizeof(float), cudaMemcpyDeviceToHost);
//         cudaMemcpy(smids_h[i], smids_d[i], smids_size, cudaMemcpyDeviceToHost);
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

    gemmKernel<<<blocksGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, alpha, beta, M, N, K, smids_d);

    return;
}
