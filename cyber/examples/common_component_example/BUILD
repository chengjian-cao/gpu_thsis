load("//tools:cpplint.bzl", "cpplint")
load("//tools:apollo_package.bzl", "apollo_package", "apollo_cc_library", "apollo_cc_binary", "apollo_component")
load("//third_party/gpus:common.bzl", "gpu_library", "if_cuda", "if_rocm")

package(default_visibility = ["//visibility:public"])

apollo_component(
    name = "libcommon_component_example.so",
    srcs = ["common_component_example.cc"],
    hdrs = ["common_component_example.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
        "//cyber/croutine:cyber_croutine",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        ":neuralnetwork_cuda",
    ],
)

apollo_cc_binary(
    name = "channel_test_writer",
    srcs = ["channel_test_writer.cc"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
        "@local_config_cuda//cuda:cudart",
    ],
)

apollo_cc_binary(
    name = "channel_prediction_writer",
    srcs = ["channel_prediction_writer.cc"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
        "@local_config_cuda//cuda:cudart",
    ],
)

# gpu_library(
#     name = "kernel_cuda",
#     srcs = ["kernel.cu"],
#     hdrs = ["kernel.h"],
#     deps = [
        
#     ] + if_cuda([
#         "@local_config_cuda//cuda:cudart",
#     ]) + if_rocm([
#         "@local_config_rocm//rocm:hip",
#     ]),
# )

gpu_library(
    name = "neuralnetwork_cuda",
    srcs = ["NeuralNetwork.cu"],
    hdrs = ["NeuralNetwork.h"],
    deps = [
        "//cyber",
    ] + if_cuda([
        "@local_config_cuda//cuda:cudart",
    ]) + if_rocm([
        "@local_config_rocm//rocm:hip",
    ]),
)

filegroup(
    name = "conf",
    srcs = [
        ":common.dag",
        ":common.launch",
    ],
)

apollo_package()

cpplint()
