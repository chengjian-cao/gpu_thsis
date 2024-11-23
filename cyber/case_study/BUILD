load("//tools:cpplint.bzl", "cpplint")
load("//tools:apollo_package.bzl", "apollo_package", "apollo_cc_library", "apollo_cc_binary", "apollo_component")
load("//third_party/gpus:common.bzl", "gpu_library", "if_cuda", "if_rocm")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "case_study",
    srcs = glob([
        "conf/**",
        "dag/**",
        "data/**",
        "launch/**",
    ]),
)

apollo_component(
    name = "lib_v1.so",
    srcs = ["v1.cc"],
    hdrs = ["v1.h"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
    ],
    visibility = ["//visibility:private"],
)

apollo_component(
    name = "lib_v2.so",
    srcs = ["v2.cc"],
    hdrs = ["v2.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
        "//cyber/croutine:cyber_croutine",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        ":neuralnetwork_cuda",
        ":matrix_multiply_cuda",
    ],
)

apollo_component(
    name = "lib_v3.so",
    srcs = ["v3.cc"],
    hdrs = ["v3.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
        # "//cyber/croutine:cyber_croutine",
        # "@local_config_cuda//cuda:cuda_headers",
        # "@local_config_cuda//cuda:cudart",
        # ":neuralnetwork_cuda",
        ":MatrixOperations",
    ],
)

apollo_component(
    name = "lib_v4.so",
    srcs = ["v4.cc"],
    hdrs = ["v4.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
        "//cyber/croutine:cyber_croutine",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        ":neuralnetwork_cuda",
        ":MatrixOperations",
        ":matrix_multiply_cuda",
    ],
)

apollo_component(
    name = "lib_v5.so",
    srcs = ["v5.cc"],
    hdrs = ["v5.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
        # "//cyber/croutine:cyber_croutine",
        # "@local_config_cuda//cuda:cuda_headers",
        # "@local_config_cuda//cuda:cudart",
        # ":neuralnetwork_cuda",
        ":MatrixOperations",
    ],
)

apollo_component(
    name = "lib_v6.so",
    srcs = ["v6.cc"],
    hdrs = ["v6.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
        # "//cyber/croutine:cyber_croutine",
        # "@local_config_cuda//cuda:cuda_headers",
        # "@local_config_cuda//cuda:cudart",
        # ":neuralnetwork_cuda",
        ":MatrixOperations",
        ":matrix_multiply_cuda",
    ],
)

apollo_component(
    name = "lib_v7.so",
    srcs = ["v7.cc"],
    hdrs = ["v7.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
        "//cyber/croutine:cyber_croutine",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        ":neuralnetwork_cuda",
        ":MatrixOperations",
        ":matrix_multiply_cuda",
    ],
)

apollo_component(
    name = "lib_v7_fusion.so",
    srcs = ["v7_fusion.cc"],
    hdrs = ["v7_fusion.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
        "//cyber/croutine:cyber_croutine",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        ":neuralnetwork_cuda",
        ":MatrixOperations",
        ":matrix_multiply_cuda",
    ],
)

apollo_component(
    name = "lib_v8_fusion.so",
    srcs = ["v8_fusion.cc"],
    hdrs = ["v8_fusion.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
        # "//cyber/croutine:cyber_croutine",
        # "@local_config_cuda//cuda:cuda_headers",
        # "@local_config_cuda//cuda:cudart",
        # ":neuralnetwork_cuda",
    ],
)

apollo_component(
    name = "lib_v8.so",
    srcs = ["v8.cc"],
    hdrs = ["v8.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
        # "//cyber/croutine:cyber_croutine",
        # "@local_config_cuda//cuda:cuda_headers",
        # "@local_config_cuda//cuda:cudart",
        # ":neuralnetwork_cuda",
    ],
)

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

gpu_library(
    name = "matrix_multiply_cuda",
    srcs = ["matrix_multiply.cu"],
    hdrs = ["matrix_multiply.h"],
    deps = [
        "//cyber",
    ] + if_cuda([
        "@local_config_cuda//cuda:cudart",
    ]) + if_rocm([
        "@local_config_rocm//rocm:hip",
    ]),
)

apollo_cc_library(
    name = "MatrixOperations",
    srcs = ["MatrixOperations.cc"],
    hdrs = ["MatrixOperations.h"],
    deps = [
        # "//cyber",
        # "//cyber/examples/proto:examples_cc_proto",
        # "@local_config_cuda//cuda:cudart",
    ],
)




apollo_package()

cpplint()