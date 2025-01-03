#include <memory>
#include <cmath> // 确保包含 <cmath> 头文件以使用 std::abs
#include "cyber/component/component.h"
#include "cyber/examples/proto/examples.pb.h"
#include "cyber/croutine/croutine.h"  // 用于获取当前的 CUDA 流
#include <cuda_runtime.h>
#include "MatrixOperations.h"
#include "cyber/time/clock.h"
using apollo::cyber::Clock;
using apollo::cyber::Component;
using apollo::cyber::ComponentBase;
using apollo::cyber::examples::proto::Driver;
using apollo::cyber::Writer;

class v8_component : public Component<Driver, Driver,Driver,Driver>  {
 public:
  bool Init() override;
  bool Proc(const std::shared_ptr<Driver>& msg0,
    const std::shared_ptr<Driver>& msg1,
    const std::shared_ptr<Driver>& msg2,
    const std::shared_ptr<Driver>& msg3
            ) override;
 private:
  std::shared_ptr<Writer<Driver>> driver_writer_ = nullptr;
  static int sync_msg_count; // 计数器
  static double total_processing_time; // 累计处理时间
};
CYBER_REGISTER_COMPONENT(v8_component)