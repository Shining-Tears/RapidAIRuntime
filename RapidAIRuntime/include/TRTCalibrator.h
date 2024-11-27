#pragma once
// tensorrt
#include <NvInfer.h>
#include <NvOnnxParser.h>

// cuda
#include <cuda_runtime_api.h>

// c++
#include <string>

enum class ChannelType: int {
    None = 0, SwapRB = 1
}

enum class CaliAlgorithm: int {
    MinMax = 0, Entpory = 1;
}

struct CaliNorm {
    int batch_size = 1;
    int infer_channel = 3;
    int patch_width = 640;
    int patch_height = 640;
    
    float mean[3] = {0.f, 0.f, 0.f };
    float std[3] = {1.f, 1.f, 1.f };
    
    ChannelType channel_type = ChannelType::SwapRB;
    CaliAlgorithm cali_algorithm = CaliAlgorithm::MinMax;
}

class TRTCalibrator: public nvinfer1::IInt8Calibrator {

    int mbatch_index_;
    int mbatch_max_index_;
    std::string mcali_path_;
    std::string mcache_path_;
    CaliNorm mcali_norm_;
public:

    TRTCalibrator(const std::string& cali_path, const std::string& cache_path, 
    const CaliNorm& cali_norm)

    nvinfer1::CalibrationAlgoType getAlgorithm() noexcept override;

    int getBatchSize() const noexcept override;

    bool getBatch(void* bindings[], const char* name[], int nbBindings) noexcept override;

    const void* readCalibrationCache(size_t& length) noexcept override;

    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

    ~TRTCalibrator();
}