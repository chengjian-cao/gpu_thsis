#include "matrix_multi.h"
#include "libsmctrl.h"

#define BLOCK_SIZE 16

__global__ void gemmKernel(float *A, float *B, float *C, float alpha, float beta, int M, int N, int K, uint8_t *smids){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M && col < N){
        // get sm id
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
void initMatrix(float *matrix, int M, int N){
    for(int i=0; i<M*N; i++){
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
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

void matrix_multi(int n, std::string name_component, uint64_t mask_1,cudaStream_t stream){

    int M=1024, N=n, K=1024;
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // limit SM usage    
    
    uint64_t mask = mask_1;  
    mask = ~mask;
  
    
    // record kernel execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *h_A = new float[M*K];
    float *h_B = new float[K*N];
    float *h_C = new float[M*N];
    
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);
    initMatrix(h_C, M, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));


    cudaStream_t stream1;
    // cudaStreamCreate(&stream1);
    stream1 = stream;
    
    cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksGrid((N+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE);
    int smids_size = blocksGrid.x * blocksGrid.y * sizeof(uint8_t);
    uint8_t *smids_d;
    cudaMalloc(&smids_d, smids_size);
    uint8_t * smids_h = new uint8_t[smids_size];
    

    // 获取当前时间作为核函数启动前的时间
    apollo::cyber::Time now_time = apollo::cyber::Time::Now();
    double start_kernel = now_time.ToMillisecond();
    cudaEventRecord(start);

    // 启动核函数
    libsmctrl_set_global_mask(mask);
    gemmKernel<<<blocksGrid, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C, alpha, beta, M, N, K, smids_d);

    // 获取当前时间作为核函数执行后的时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    apollo::cyber::Time end_time = apollo::cyber::Time::Now();
    double end_kernel = end_time.ToMillisecond();

    AINFO << name_component << " kernel launch started at " << (long long)start_kernel << " ms," << " ended at " << (long long)end_kernel << " ms."<< " The processing time is "<<end_kernel-start_kernel<<" ms.";

    cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(smids_h, smids_d, smids_size, cudaMemcpyDeviceToHost);

    int *uniq_ids = new int[smids_size];
    int uniq_counts = count_unique(smids_h, smids_size, uniq_ids);
    std::cout << "The number of different SMs used in "<< name_component <<": " << uniq_counts << std::endl;

    for(int i=0; i<uniq_counts; i++){
        std::cout << uniq_ids[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(smids_d);


    delete [] h_A;
    delete [] h_B;
    delete [] h_C;
    delete [] smids_h;
    delete [] uniq_ids;

    return;

}