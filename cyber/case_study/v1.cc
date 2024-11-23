#include "cyber/case_study/v1.h"
#include "cyber/class_loader/class_loader.h"
#include "cyber/component/component.h"
#include "cyber/examples/proto/examples.pb.h"
#include <cuda_runtime.h>

bool v1_timer_component::Init() {
  driver_writer_ = node_->CreateWriter<Driver>("/apollo/v1");
  return true;
}

bool v1_timer_component::Proc() {
    apollo::cyber::Time now_time = apollo::cyber::Time::Now();
    double current_time = now_time.ToMillisecond();

    auto out_msg = std::make_shared<Driver>();
    out_msg->set_start_time(current_time);
    out_msg->set_timestamp(current_time);
   
    // driver_writer_->Write(out_msg);

    static int i = 0;
    // auto out_msg = std::make_shared<Driver>();
    out_msg->set_msg_id(i++);
    driver_writer_->Write(out_msg);
    //  AINFO << "================================================";
    // AINFO << "v1_timer_component_example: Write drivermsg->"
    //       << out_msg->ShortDebugString();
    return true;
}
