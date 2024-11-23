#include <memory>

#include "cyber/component/component.h"
#include "cyber/examples/proto/examples.pb.h"
#include "cyber/component/timer_component.h"
#include "cyber/class_loader/class_loader.h"

using apollo::cyber::Component;
using apollo::cyber::ComponentBase;
using apollo::cyber::examples::proto::Driver;
using apollo::cyber::Writer;
using apollo::cyber::Time;

class v8_fusion_component : public Component<Driver, Driver,Driver,Driver> {
 public:
  bool Init() override;
  
  bool Proc(const std::shared_ptr<Driver>& msg0,
    const std::shared_ptr<Driver>& msg1,
    const std::shared_ptr<Driver>& msg2,
    const std::shared_ptr<Driver>& msg3
            ) override;
 private:
  std::shared_ptr<Writer<Driver>> driver_writer_ = nullptr;
  // 检查并处理同步的消息对
  void processSynchronizedMessages();
    // 使用哈希表缓冲消息
  std::unordered_map<int, std::shared_ptr<Driver>> msg0_buffer_;
  std::unordered_map<int, std::shared_ptr<Driver>> msg1_buffer_;
  std::unordered_map<int, std::shared_ptr<Driver>> msg2_buffer_;
  std::unordered_map<int, std::shared_ptr<Driver>> msg3_buffer_;

};
CYBER_REGISTER_COMPONENT(v8_fusion_component)