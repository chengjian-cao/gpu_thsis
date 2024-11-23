#include "cyber/case_study/v7_fusion.h"

bool v7_fusion_component::Init() {
//   AINFO << "v7_fusion_component init";
  driver_writer_ = node_->CreateWriter<Driver>("/apollo/v7_fusion");
  return true;
}

bool v7_fusion_component::Proc(const std::shared_ptr<Driver>& msg0, const std::shared_ptr<Driver>& msg1) {
    // 缓存消息到对应的缓冲区
    v5_buffer_[msg0->msg_id()] = msg0;
    v6_buffer_[msg1->msg_id()] = msg1;

    // 只调用一次同步检查
    processSynchronizedMessages();
    return true;
}
void v7_fusion_component::processSynchronizedMessages() {
    // 遍历 v5_buffer_ 中的所有 msg_id
    for (auto it = v5_buffer_.begin(); it != v5_buffer_.end();) {
        int msg_id = it->first;

        // 检查 v6_buffer_ 是否有相同的 msg_id
        if (v6_buffer_.count(msg_id) > 0) {
            auto v5_msg = v5_buffer_[msg_id];
            auto v6_msg = v6_buffer_[msg_id];

            // 计算延迟
            double v5_timestamp = v5_msg->timestamp();
            double v6_timestamp = v6_msg->timestamp();
            double v7_receive_time = std::max(v5_timestamp, v6_timestamp);

            // 创建输出消息
            auto out_msg = std::make_shared<Driver>();
            out_msg->set_msg_id(msg_id);
            out_msg->set_timestamp(v7_receive_time);
            out_msg->set_start_time(v5_msg->start_time());
            driver_writer_->Write(out_msg);

            // 移除已处理的消息
            v5_buffer_.erase(msg_id);
            v6_buffer_.erase(msg_id);

            // 重置迭代器，避免迭代器失效
            it = v5_buffer_.begin();
        } else {
            // 如果未找到同步的消息，对应 msg_id 保留在缓冲区中
            ++it;
        }
    }
}
