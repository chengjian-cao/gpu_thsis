#include "cyber/case_study/v8_fusion.h"

bool v8_fusion_component::Init() {
  driver_writer_ = node_->CreateWriter<Driver>("/apollo/v8_fusion");
  return true;
}
// , const std::shared_ptr<Driver>& msg1, const std::shared_ptr<Driver>& msg2, const std::shared_ptr<Driver>& msg3
bool v8_fusion_component::Proc(const std::shared_ptr<Driver>& msg0, const std::shared_ptr<Driver>& msg1, 
                        const std::shared_ptr<Driver>& msg2, const std::shared_ptr<Driver>& msg3) {
    // 将每个消息存储到各自的缓冲区
    msg0_buffer_[msg0->msg_id()] = msg0;
    msg1_buffer_[msg1->msg_id()] = msg1;
    msg2_buffer_[msg2->msg_id()] = msg2;
    msg3_buffer_[msg3->msg_id()] = msg3;

    // 处理同步消息
    processSynchronizedMessages();

    return true;
}

void v8_fusion_component::processSynchronizedMessages() {
    // 遍历所有缓冲区，找出可以同步处理的 `msg_id`
    for (auto it = msg0_buffer_.begin(); it != msg0_buffer_.end();) {
        int msg_id = it->first;

        // 检查是否所有缓冲区都有相同的 `msg_id`
        if (msg1_buffer_.count(msg_id) > 0 &&
            msg2_buffer_.count(msg_id) > 0 &&
            msg3_buffer_.count(msg_id) > 0) {
            // 获取同步的消息
            auto msg0 = msg0_buffer_[msg_id];
            auto msg1 = msg1_buffer_[msg_id];
            auto msg2 = msg2_buffer_[msg_id];
            auto msg3 = msg3_buffer_[msg_id];

            // 计算同步时间戳
            double v8_receive_time = std::max({msg0->timestamp(), msg1->timestamp(),
                                             msg2->timestamp(), msg3->timestamp()});

            // 创建输出消息
            auto out_msg = std::make_shared<Driver>();
            out_msg->set_msg_id(msg_id);
            out_msg->set_timestamp(v8_receive_time);
            out_msg->set_start_time(msg0->start_time());
            driver_writer_->Write(out_msg);

            // 从缓冲区中移除已处理的消息
            msg0_buffer_.erase(msg_id);
            msg1_buffer_.erase(msg_id);
            msg2_buffer_.erase(msg_id);
            msg3_buffer_.erase(msg_id);

            // 移动到下一个迭代
            it = msg0_buffer_.begin();
        } else {
            // 如果无法找到同步的消息，跳过
            ++it;
        }
    }

}
