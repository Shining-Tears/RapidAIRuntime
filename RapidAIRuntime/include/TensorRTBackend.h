#pragma once
// tensorrt
#include <NvInfer.h>
#include <NvOnnxParser.h>

// cuda
#include <cuda_runtime_api.h>

// C++
#include <stdio.h>
#include <mutex>
#include <thread>

#include "utility.h"

namespace TRTBackend {

enum class EnginePrecision {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2
};

enum class Optimizationlevel {
    Level0 = 0,
    Level1 = 1,
    Level2 = 2,
    Level3 = 3,
    Level4 = 4,
    Level5 = 5
};

struct EngineDynamicShape {
    nvinfer1::Dims minInputShape;
    nvinfer1::Dims maxInputShape;
    nvinfer1::Dims optInputShape;
};

class TRTLogger : public nvinfer1::ILogger {
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override;
};

class TRTObjectProvider {

    static TRTLogger mlogger_;
    static nvinfer1::IBuilder* mbuilder_;
    static nvinfer1::IRuntime* mruntime_;
    static std::once_flag builder_flag_;
    static std::once_flag runtime_flag_;
public:

    static TRTLogger& get_trt_logger();

    static nvinfer1::IBuilder* get_trt_builder();

    static nvinfer1::IRuntime* get_trt_runtime();
};

class TRTEngine {

    nvinfer1::ICudaEngine* mengine_ = nullptr;
    nvinfer1::IBuilderConfig* mconfig_ = nullptr;
    nvinfer1::INetworkDefinition* mnetwork_ = nullptr;
    nvonnxparser::IParser* mparser_ = nullptr;
    nvinfer1::IOptimizationProfile* mprofile_ = nullptr;
    nvinfer1::IHostMemory* mengine_data_ = nullptr;
    TRTObjectProvider mprovider_;

    ~TRTEngine() noexcept;
    void init_members(std::string& onnx_path);
    void build_engine(EnginePrecision& engine_precision, Optimizationlevel& optimize_level);
public:
    // 反序列化已有的TRT文件
    TRTEngine(std::string trt_path);

    // ONNX转换静态输入的TRT推理引擎
    TRTEngine(std::string onnx_path, EnginePrecision engine_precision, 
    Optimizationlevel optimize_level);

    // ONNX转换动态输入的TRT推理引擎
    TRTEngine(std::string onnx_path, EnginePrecision engine_precision, 
    Optimizationlevel optimize_level, EngineDynamicShape dynamic_shape);

    TRTEngine(std::string onnx_path, EnginePrecision engine_precision, 
    Optimizationlevel optimize_level, std::vector<EngineDynamicShape> dynamic_shapes);

    void save_engine(std::string save_path);
};
}