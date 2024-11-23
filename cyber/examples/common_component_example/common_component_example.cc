/******************************************************************************
 * Copyright 2018 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/
#include "cyber/examples/common_component_example/common_component_example.h"
#include "cyber/croutine/croutine.h"  // 用于获取当前的 CUDA 流
#include <cuda_runtime.h>
#include "NeuralNetwork.h"

bool CommonComponentSample::Init() {
  AINFO << "Commontest component init";
  return true;
}
using apollo::cyber::croutine::CRoutine;

bool CommonComponentSample::Proc(const std::shared_ptr<Driver>& msg0,
                                 const std::shared_ptr<Driver>& msg1) {
  // 获取当前 CUDA 流
  cudaStream_t stream = CRoutine::GetCurrentStream();
  if (stream != nullptr) {
      AINFO << "CUDA stream is available is " << stream;
  }
    int num_layers = 3;       // 神经网络层数
    int input_size = 1024;    // 输入和输出的大小

    // 创建神经网络实例
    NeuralNetwork nn(num_layers, stream);

    // 分配主机内存
    float* input = new float[input_size];
    float* output = new float[input_size];

    // 初始化输入数据（这里以随机数为例）
    for (int i = 0; i < input_size; ++i) {
        input[i] = static_cast<float>(i % 100) / 100.0f;
    }

    // 执行前向传播
    nn.forward(input, output, input_size);

    // // 输出部分结果（例如前10个值）
    // AINFO << "Output:" ;
    // for (int i = 0; i < 10; ++i) {
    //     AINFO  << output[i];
    // }
    

    // 释放资源
    delete[] input;
    delete[] output;

  
  AINFO << "Start common component Proc [" << msg0->msg_id() << "] ["
        << msg1->msg_id() << "]";
  return true;
}
