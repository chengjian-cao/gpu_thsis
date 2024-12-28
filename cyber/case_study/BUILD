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
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    visibility = ["//visibility:private"],
)

apollo_component(
    name = "lib_test.so",
    srcs = ["test.cc"],
    hdrs = ["test.h"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
        "//cyber/croutine:cyber_croutine",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        ":gemm_multi_streams",
        ":MatrixOperations",
        ":matrix_multi",
    ],
    visibility = ["//visibility:private"],
)

apollo_component(
    name = "lib_test_limited.so",
    srcs = ["test_limited.cc"],
    hdrs = ["test_limited.h"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
        "//cyber/croutine:cyber_croutine",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        ":gemm_multi_streams",
        ":MatrixOperations",
        ":matrix_multi",
        ":limited_blocks_kernel"
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
        ":gemm_multi_streams",
        ":MatrixOperations",
        ":matrix_multi",
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
        ":MatrixOperations",
        ":gemm_multi_streams",
        ":matrix_multi",
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
        ":gemm_multi_streams",
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
        ":MatrixOperations",
        ":gemm_multi_streams",
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
    ],
)


gpu_library(
    name = "gemm_multi_streams",
    srcs = ["gemm_multi_streams.cu"],
    hdrs = ["gemm_multi_streams.h"],
    deps = [
        "//cyber",
        ":smctrl",
    ] + if_cuda([
        "@local_config_cuda//cuda:cudart",
    ]) + if_rocm([
        "@local_config_rocm//rocm:hip",
    ]),
)

gpu_library(
    name = "matrix_multi",
    srcs = ["matrix_multi.cu"],
    hdrs = ["matrix_multi.h"],
    deps = [
        "//cyber",
        ":smctrl",
    ] + if_cuda([
        "@local_config_cuda//cuda:cudart",
    ]) + if_rocm([
        "@local_config_rocm//rocm:hip",
    ]),
)

gpu_library(
    name = "limited_blocks_kernel",
    srcs = ["limited_blocks_kernel.cu"],
    hdrs = ["limited_blocks_kernel.h"],
    deps = [
        "//cyber",
        ":smctrl",
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


apollo_cc_library(
    name = "smctrl",
    srcs = ["libsmctrl.c"],
    hdrs = ["libsmctrl.h"],
    deps = [
        # "//cyber",
        # "//cyber/examples/proto:examples_cc_proto",
        "@local_config_cuda//cuda:cudart",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_driver",  # 添加 CUDA 驱动库
    ],
)

apollo_cc_binary(
    name = "channel_writer",
    srcs = ["write_sig.cc"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
        "@local_config_cuda//cuda:cudart",
    ],
)

apollo_cc_binary(
    name = "libsmctrl_test_gpc_info",
    srcs = [
        "libsmctrl_test_gpc_info.c",
        "testbench.h",
    ],
    deps = [
        ":smctrl",        
        "@local_config_cuda//cuda:cudart",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_driver",  # 添加 CUDA 驱动库
    ],
)

gpu_library(
    name = "smctrl_test_mask_shared",
    srcs = ["libsmctrl_test_mask_shared.cu",
            "testbench.h",
            "libsmctrl_test_mask_shared.h"],
    deps = [
        ":smctrl", 
    ] + if_cuda([
        "@local_config_cuda//cuda:cudart",
    ]) + if_rocm([
        "@local_config_rocm//rocm:hip",
    ]),
)

apollo_cc_binary(
    name = "libsmctrl_test_global_mask",
    srcs = ["libsmctrl_test_global_mask.c",
            "libsmctrl_test_mask_shared.h"],
    deps = [
        ":smctrl",        
        ":smctrl_test_mask_shared",
        "@local_config_cuda//cuda:cudart",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_driver",  # 添加 CUDA 驱动库
    ],
)




apollo_package()

cpplint()