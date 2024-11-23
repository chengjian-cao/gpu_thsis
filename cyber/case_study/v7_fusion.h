#include <memory>
#include <cmath> // 确保包含 <cmath> 头文件以使用 std::abs
#include "cyber/component/component.h"
#include "cyber/examples/proto/examples.pb.h"
#include "MatrixOperations.h"
#include "NeuralNetwork.h"
#include "cyber/croutine/croutine.h"  // 用于获取当前的 CUDA 流
#include <cuda_runtime.h>
using apollo::cyber::Component;
using apollo::cyber::ComponentBase;
using apollo::cyber::examples::proto::Driver;
using apollo::cyber::Writer;

class v7_fusion_component : public Component<Driver, Driver> {
 public:
  bool Init() override;
  bool Proc(const std::shared_ptr<Driver>& msg0,
            const std::shared_ptr<Driver>& msg1) override;
 private:
  std::shared_ptr<Writer<Driver>> driver_writer_ = nullptr;
    // 使用哈希表缓冲来自 v5 和 v6 的消息
    std::unordered_map<int, std::shared_ptr<Driver>> v5_buffer_;
    std::unordered_map<int, std::shared_ptr<Driver>> v6_buffer_;

    // 检查并处理同步的消息对
    void processSynchronizedMessages();
};
CYBER_REGISTER_COMPONENT(v7_fusion_component)