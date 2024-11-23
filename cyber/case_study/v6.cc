#include "cyber/case_study/v6.h"
#include "MatrixOperations.h"
bool v6_component::Init() {
//   AINFO << "v6_component init";
  driver_writer_ = node_->CreateWriter<Driver>("/apollo/v6");
  return true;
}

bool v6_component::Proc(const std::shared_ptr<Driver>& msg0) {
    auto out_msg = std::make_shared<Driver>();
    // out_msg->set_msg_id(msg0->msg_id());
    apollo::cyber::Time now_time = apollo::cyber::Time::Now();
    double start_time = now_time.ToMillisecond();

    int sizeA = 32; // 第一矩阵的行数和第二矩阵的列数
    int sizeB = 32; // 第一矩阵的列数和第二矩阵的行数
    // 执行矩阵生成与相乘
    Matrix result = MatrixOperations::generateAndMultiplyMatrices(sizeA, sizeB);

    now_time = apollo::cyber::Time::Now();
    double end_time = now_time.ToMillisecond();
    // AINFO << "v6 matrix time is "<<end_time-start_time;

    out_msg->set_timestamp(msg0->start_time()+end_time-start_time);
    out_msg->set_msg_id(msg0->msg_id());
    out_msg->set_start_time(msg0->start_time());
    driver_writer_->Write(out_msg);
    // long long int_start_time = static_cast<long long>(end_time);
    // std::string s = std::to_string(int_start_time);

    return true;
}