#include <memory>
#include "cyber/component/component.h"
#include "cyber/examples/proto/examples.pb.h"
#include "gemm_multi_streams.h"
#include "MatrixOperations.h"
#include "cyber/time/clock.h"
using apollo::cyber::Clock;
using apollo::cyber::Component;
using apollo::cyber::ComponentBase;
using apollo::cyber::examples::proto::Driver;
using apollo::cyber::Writer;

class v6_component : public Component<Driver> {
 public:
  bool Init() override;
  bool Proc(const std::shared_ptr<Driver>& msg0) override;

 private:
  std::shared_ptr<Writer<Driver>> driver_writer_ = nullptr;
};
CYBER_REGISTER_COMPONENT(v6_component)