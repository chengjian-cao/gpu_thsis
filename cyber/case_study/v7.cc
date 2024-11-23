#include "cyber/case_study/v7.h"

bool v7_component::Init() {
  driver_writer_ = node_->CreateWriter<Driver>("/apollo/v7");
  return true;
}

bool v7_component::Proc(const std::shared_ptr<Driver>& msg0) {

    apollo::cyber::Time now_time = apollo::cyber::Time::Now();
    double start_time = now_time.ToMillisecond();
    
    // int sizeA = 64; // 第一矩阵的行数和第二矩阵的列数
    // int sizeB = 64; // 第一矩阵的列数和第二矩阵的行数
    // // 执行矩阵生成与相乘
    // Matrix result = MatrixOperations::generateAndMultiplyMatrices(sizeA, sizeB);

    //获取当前 CUDA 流
    #ifdef USE_NULLPTR_STREAM
    cudaStream_t stream = nullptr;
    #else
    cudaStream_t stream = apollo::cyber::croutine::CRoutine::GetCurrentStream();
    if (stream != nullptr) {
      AINFO << "CUDA stream is available is " << stream;
    }
    #endif
    // if (stream != nullptr) {
    //     AINFO << "CUDA stream is available is " << stream;
    // }
    // int num_layers = 10;       // 神经网络层数
    // int input_size = 1024;    // 输入和输出的大小

    // // 创建神经网络实例
    // NeuralNetwork nn(num_layers, stream);

    // // 分配主机内存
    // float* input = new float[input_size];
    // float* output = new float[input_size];

    // // 初始化输入数据（这里以随机数为例）
    // for (int i = 0; i < input_size; ++i) {
    //     input[i] = static_cast<float>(i % 100) / 100.0f;
    // }

    // // 执行前向传播
    // nn.forward(input, output, input_size);

    // // 释放资源
    // delete[] input;
    // delete[] output;

    int N = 2048;
    matrixMultiplyTest(N, stream, "v7_component");

    now_time = apollo::cyber::Time::Now();
    double end_time = now_time.ToMillisecond();

    // 创建输出消息
    auto out_msg = std::make_shared<Driver>();
    out_msg->set_msg_id(msg0->msg_id());
    out_msg->set_timestamp(msg0->timestamp()+ end_time - start_time);
    out_msg->set_start_time(msg0->start_time());
    driver_writer_->Write(out_msg);

    return true;
}
