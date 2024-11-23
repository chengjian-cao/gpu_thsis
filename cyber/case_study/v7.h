#include <memory>
#include <cmath> // 确保包含 <cmath> 头文件以使用 std::abs
#include "cyber/component/component.h"
#include "cyber/examples/proto/examples.pb.h"
#include "MatrixOperations.h"
#include "NeuralNetwork.h"
#include "matrix_multiply.h"
#include "cyber/croutine/croutine.h"  // 用于获取当前的 CUDA 流
#include <cuda_runtime.h>
using apollo::cyber::Component;
using apollo::cyber::ComponentBase;
using apollo::cyber::examples::proto::Driver;
using apollo::cyber::Writer;

class v7_component : public Component<Driver> {
 public:
  bool Init() override;
  bool Proc(const std::shared_ptr<Driver>& msg0) override;
 private:
  std::shared_ptr<Writer<Driver>> driver_writer_ = nullptr;
};
CYBER_REGISTER_COMPONENT(v7_component)