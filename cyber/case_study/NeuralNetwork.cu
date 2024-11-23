#include "NeuralNetwork.h"
#include <cuda_runtime.h>
#include <algorithm> // for std::max
#include <random>
#include "cyber/cyber.h"
// 简单的全连接层 CUDA 核函数实现
__global__ void fullyConnectedLayer(
    float* input, float* output, float* weights, float* biases, int input_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size) {
        float sum = biases[idx];
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * weights[i * input_size + idx];
        }
        // 激活函数（ReLU）
        output[idx] = fmaxf(0.0f, sum);
    }
}

// 构造函数实现
NeuralNetwork::NeuralNetwork(int num_layers, cudaStream_t stream)
    : num_layers_(num_layers), stream_(stream), d_input_(nullptr), d_output_(nullptr)
{
    // 初始化权重和偏置将在 forward 函数中完成
}

// 析构函数实现
NeuralNetwork::~NeuralNetwork()
{
    // 释放设备内存
    if (d_input_) cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);

    // 释放每一层的权重和偏置
    for (auto& w : weights_) {
        if (w) cudaFree(w);
    }
    for (auto& b : biases_) {
        if (b) cudaFree(b);
    }
}

// 初始化权重和偏置
void NeuralNetwork::initializeWeights(int input_size)
{
    // 使用随机数生成器初始化权重和偏置
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-0.1f, 0.1f);

    for (int i = 0; i < num_layers_; ++i) {
        float* h_weight = new float[input_size * input_size];
        float* h_bias = new float[input_size];

        // 初始化权重和偏置
        for (int j = 0; j < input_size * input_size; ++j) {
            h_weight[j] = distribution(generator);
        }
        for (int j = 0; j < input_size; ++j) {
            h_bias[j] = distribution(generator);
        }

        float* d_weight;
        float* d_bias;

        // 为权重矩阵和偏置向量分配设备内存
        cudaMalloc(&d_weight, input_size * input_size * sizeof(float));
        cudaMalloc(&d_bias, input_size * sizeof(float));

        // 将数据从主机拷贝到设备
        cudaMemcpyAsync(d_weight, h_weight, input_size * input_size * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_bias, h_bias, input_size * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);

        // 同步 CUDA 流
        cudaStreamSynchronize(stream_);

        // 释放主机内存
        delete[] h_weight;
        delete[] h_bias;

        weights_.push_back(d_weight);
        biases_.push_back(d_bias);
    }
}

// 前向传播函数实现，添加了时间测量
void NeuralNetwork::forward(float* input, float* output, int input_size)
{
    int layer_size = input_size;
    int threadsPerBlock = 256;
    int blocksPerGrid = (layer_size + threadsPerBlock - 1) / threadsPerBlock;

    // 如果权重和偏置未初始化，则初始化
    if (weights_.empty() || biases_.empty()) {
        initializeWeights(input_size);
    }

    // 将输入数据拷贝到设备
    cudaMalloc(&d_input_, input_size * sizeof(float));
    cudaMemcpyAsync(d_input_, input, input_size * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);

    float* d_current_input = d_input_;
    float* d_current_output = nullptr;

    // 创建 CUDA 事件
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // 记录开始事件
    cudaEventRecord(start_event, stream_);

    // 遍历每一层
    for (int i = 0; i < num_layers_; ++i) {
        cudaMalloc(&d_current_output, layer_size * sizeof(float));

        // 调用全连接层的 CUDA 核函数
        fullyConnectedLayer<<<blocksPerGrid, threadsPerBlock, 0, stream_>>>(
            d_current_input, d_current_output, weights_[i], biases_[i], layer_size);

        // 同步以确保当前层的计算完成
        cudaStreamSynchronize(stream_);

        // 释放上一层的输入
        if (i != 0) {
            cudaFree(d_current_input);
        }
        d_current_input = d_current_output;
    }

    // 记录结束事件
    cudaEventRecord(stop_event, stream_);

    // 确保事件已完成
    cudaEventSynchronize(stop_event);

    // 计算时间差
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);

    // 输出计算时间
    AINFO << "Forward pass took " << milliseconds << " milliseconds." ;

    // 将最终输出拷贝回主机内存
    cudaMemcpyAsync(output, d_current_output, layer_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // 释放设备内存
    cudaFree(d_current_input);

    // 销毁 CUDA 事件
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
}
