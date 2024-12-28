#include <memory>
#include "cyber/component/component.h"
#include "cyber/examples/proto/examples.pb.h"
#include "cyber/time/clock.h"
#include "cyber/croutine/croutine.h"  // 用于获取当前的 CUDA 流
#include <cuda_runtime.h>
#include "gemm_multi_streams.h"
#include "matrix_multi.h"
#include "MatrixOperations.h"

using apollo::cyber::Component;
using apollo::cyber::ComponentBase;
using apollo::cyber::examples::proto::Driver;
using apollo::cyber::Writer;
using apollo::cyber::Clock;

class v2_component : public Component<Driver> {
 public:
  bool Init() override;
  bool Proc(const std::shared_ptr<Driver>& msg0) override;

 private:
  std::shared_ptr<Writer<Driver>> driver_writer_ = nullptr;
};
CYBER_REGISTER_COMPONENT(v2_component)