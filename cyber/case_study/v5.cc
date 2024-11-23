#include "cyber/case_study/v5.h"

bool v5_component::Init() {
  // AINFO << "v5_component init";
  driver_writer_ = node_->CreateWriter<Driver>("/apollo/v5");
  return true;
}

bool v5_component::Proc(const std::shared_ptr<Driver>& msg0) {
    auto out_msg = std::make_shared<Driver>();
    // out_msg->set_msg_id(msg0->msg_id());
    apollo::cyber::Time now_time = apollo::cyber::Time::Now();
    double start_time = now_time.ToMillisecond();



    int sizeA = 128; // 第一矩阵的行数和第二矩阵的列数
    int sizeB = 128; // 第一矩阵的列数和第二矩阵的行数
    // 执行矩阵生成与相乘
    Matrix result = MatrixOperations::generateAndMultiplyMatrices(sizeA, sizeB);



  

    now_time = apollo::cyber::Time::Now();
    double end_time = now_time.ToMillisecond();
    // long long int_start_time = static_cast<long long>(start_time);
    // std::string s = std::to_string(int_start_time);
    out_msg->set_timestamp(msg0->start_time()+ end_time-start_time);
    out_msg->set_msg_id(msg0->msg_id());
    out_msg->set_start_time(msg0->start_time());
    driver_writer_->Write(out_msg);
    // AINFO << "v5_component: Processed drivermsg->" << msg0->ShortDebugString()
    //       << " with new timestamp " << s;
    return true;
}