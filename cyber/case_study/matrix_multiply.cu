#include "matrix_multiply.h"
#include <iostream>
#include <cuda_runtime.h>

// CUDA 核函数：计算 C = A * B
__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 当前线程对应的行
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 当前线程对应的列

    if (row < N && col < N) {
        float value = 0.0f;
        for (int i = 0; i < N; ++i) {
            value += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}

// 矩阵乘法测试函数
void matrixMultiplyTest(int N, cudaStream_t stream, std::string name_component) {
    // 矩阵大小
    size_t matrixSize = N * N * sizeof(float);

    // 分配主机内存
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    // 初始化矩阵 A 和 B（可以使用随机数）
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(i % 100); // 简单初始化
        h_B[i] = static_cast<float>((i + 1) % 100);
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, matrixSize);
    cudaMalloc(&d_B, matrixSize);
    cudaMalloc(&d_C, matrixSize);

    // 拷贝数据到设备
    cudaMemcpyAsync(d_A, h_A, matrixSize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, matrixSize, cudaMemcpyHostToDevice, stream);

    // 设置线程块和网格大小
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 获取当前时间作为核函数启动前的时间
    apollo::cyber::Time now_time = apollo::cyber::Time::Now();
    double start_kernel = now_time.ToMillisecond();


    // 启动核函数
    if (stream == nullptr) {
        matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
    } else {
        matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, N);
        cudaStreamSynchronize(stream);
    }

    // 获取当前时间作为核函数执行后的时间
    apollo::cyber::Time end_time = apollo::cyber::Time::Now();
    double end_kernel = end_time.ToMillisecond();

    // 打印核函数结束后的时间和执行时间
    if(!name_component.empty())
    AINFO << name_component << " kernel launch started at " << (long long)start_kernel << " ms," << " ended at " << (long long)end_kernel << " ms."<< " The processing time is "<<end_kernel-start_kernel<<" ms.";


    // 将结果拷贝回主机
    cudaMemcpyAsync(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost, stream);
    if (stream != nullptr) {
        cudaStreamSynchronize(stream);
    }

    // // 打印结果（仅测试较小矩阵）
    // if (N <= 4) {
    //     std::cout << "Matrix A:" << std::endl;
    //     for (int i = 0; i < N; ++i) {
    //         for (int j = 0; j < N; ++j) {
    //             std::cout << h_A[i * N + j] << " ";
    //         }
    //         std::cout << std::endl;
    //     }

    //     std::cout << "Matrix B:" << std::endl;
    //     for (int i = 0; i < N; ++i) {
    //         for (int j = 0; j < N; ++j) {
    //             std::cout << h_B[i * N + j] << " ";
    //         }
    //         std::cout << std::endl;
    //     }

    //     std::cout << "Matrix C (Result):" << std::endl;
    //     for (int i = 0; i < N; ++i) {
    //         for (int j = 0; j < N; ++j) {
    //             std::cout << h_C[i * N + j] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // } else {
    //     std::cout << "Matrix multiplication for N=" << N << " completed." << std::endl;
    // }

    // 释放内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
