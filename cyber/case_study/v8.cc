#include "cyber/case_study/v8.h"

// Define and initialize static members
int v8_component::sync_msg_count = 0;
double v8_component::total_processing_time = 0.0;

bool v8_component::Init() {
//   driver_writer_ = node_->CreateWriter<Driver>("/apollo/v7");
  return true;
}

bool v8_component::Proc(const std::shared_ptr<Driver>& msg0) {

    // 增加计数器
    sync_msg_count++;
    total_processing_time += msg0->timestamp() - msg0->start_time();

    // 打印每条消息的处理时间
    AINFO << "v8_component: Processed a message ["<< msg0->msg_id()<<"], processing time: " 
      << msg0->timestamp()-msg0->start_time() << " ms";

    // 每100次处理同步消息后，记录平均花费时间
    if (sync_msg_count >= 100) {
        double avg_time = total_processing_time / sync_msg_count;
        AINFO << "v8_component: Processed 100 synchronized messages, average processing time: "
              << avg_time << " ms";

        // 重置计数器和累计时间
        sync_msg_count = 0;
        total_processing_time = 0;
    }
    return true;
}