#include <gtest/gtest.h>
#include <fstream>
#include <cstring>
#include <memory>

#include "TensorRTBackend.h"

std::string onnx_path = "./resnet50.onnx";

TEST(test_converter, onnx_path) {
    std::ifstream file(onnx_path);

    ASSERT_TRUE(file.is_open());
}

TEST(test_converter, float32) {
    std::string save_path = "./resnet50_fp32.trt";

    TRTBackend::EnginePrecision precision = TRTBackend::EnginePrecision::FP32;
    TRTBackend::Optimizationlevel optimization_level = TRTBackend::Optimizationlevel::Level0;
    TRTBackend::TRTEngine mengine = TRTBackend::TRTEngine(onnx_path, precision, optimization_level);

    mengine.save_engine(save_path);
    std::ifstream file(save_path);

    ASSERT_TRUE(file.is_open());
}

TEST(test_converter, float16) {
    std::string save_path = "./resnet50_fp16.trt";

    TRTBackend::EnginePrecision precision = TRTBackend::EnginePrecision::FP16;
    TRTBackend::Optimizationlevel optimization_level = TRTBackend::Optimizationlevel::Level0;
    TRTBackend::TRTEngine mengine = TRTBackend::TRTEngine(onnx_path, precision, optimization_level);

    mengine.save_engine(save_path);
    std::ifstream file(save_path);

    ASSERT_TRUE(file.is_open());
}