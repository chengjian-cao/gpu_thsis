

#include <vector>
#include <cuda_runtime.h>

// 神经网络类定义
class NeuralNetwork {
public:
    // 构造函数，接受层数和 CUDA 流作为参数
    NeuralNetwork(int num_layers, cudaStream_t stream);
    ~NeuralNetwork();

    // 前向传播函数，输入和输出都是一维数组
    void forward(float* input, float* output, int input_size);

private:
    int num_layers_;                 // 神经网络的层数
    cudaStream_t stream_;            // CUDA 流
    std::vector<float*> weights_;    // 每一层的权重矩阵
    std::vector<float*> biases_;     // 每一层的偏置向量

    // 初始化权重和偏置
    void initializeWeights(int input_size);

    // 设备端输入和输出指针
    float* d_input_;
    float* d_output_;
};

// CUDA 核函数声明
__global__ void fullyConnectedLayer(
    float* input, float* output, float* weights, float* biases, int input_size);


