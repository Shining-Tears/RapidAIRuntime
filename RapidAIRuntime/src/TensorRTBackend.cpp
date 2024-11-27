#include "TensorRTBackend.h"

namespace TRTBackend {
inline const char* severity_string(nvinfer1::ILogger::Severity t) {
    switch (t) {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
    case nvinfer1::ILogger::Severity::kERROR:   return "error";
    case nvinfer1::ILogger::Severity::kWARNING: return "warning";
    case nvinfer1::ILogger::Severity::kINFO:    return "info";
    case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
    default: return "unknow";
    }
}

void TRTLogger::log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept {
    if (severity <= Severity::kINFO) {
        if (severity == Severity::kWARNING) {
            printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
        }
        else if (severity <= Severity::kERROR) {
            printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
        }
        else {
            printf("%s: %s\n", severity_string(severity), msg);
        }
    }
}

TRTLogger TRTObjectProvider::mlogger_;
std::once_flag TRTObjectProvider::builder_flag_;
std::once_flag TRTObjectProvider::runtime_flag_;
nvinfer1::IBuilder* TRTObjectProvider::mbuilder_ = nullptr;
nvinfer1::IRuntime* TRTObjectProvider::mruntime_ = nullptr;

TRTLogger& TRTObjectProvider::get_trt_logger() {
    return mlogger_;
}

nvinfer1::IBuilder* TRTObjectProvider::get_trt_builder() {
    if (mbuilder_ == nullptr) {
        std::call_once(builder_flag_, [&]() {
            mbuilder_ = nvinfer1::createInferBuilder(mlogger_);        
            uint32_t threads = std::max(std::thread::hardware_concurrency() / 2, uint32_t(1));
            mbuilder_->setMaxThreads(threads);
            mbuilder_->setMaxBatchSize(1);
        });
    }

    return mbuilder_;
}

nvinfer1::IRuntime* TRTObjectProvider::get_trt_runtime() {
    if (mruntime_ == nullptr) {
        std::call_once(runtime_flag_, [&]() {
            mruntime_ = nvinfer1::createInferRuntime(mlogger_);;
        });
    }

    return mruntime_;
}

TRTEngine::TRTEngine(std::string trt_path) {
    std::vector<unsigned char> trt_data = load_file(trt_path);
    nvinfer1::IRuntime* mruntime = mprovider_.get_trt_runtime();
    mengine_ = mruntime->deserializeCudaEngine(trt_data.data(), trt_data.size());
}

TRTEngine::TRTEngine(std::string onnx_path, EnginePrecision engine_precision, 
    Optimizationlevel optimize_level) {
    init_members(onnx_path);
    build_engine(engine_precision, optimize_level);
}

TRTEngine::TRTEngine(std::string onnx_path, EnginePrecision engine_precision, 
    Optimizationlevel optimize_level, EngineDynamicShape dynamic_shape) {
    init_members(onnx_path);

    nvinfer1::ITensor* inputTensor = mnetwork_->getInput(0);
    mprofile_->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMIN, \
        dynamic_shape.minInputShape);
    mprofile_->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMAX, \
        dynamic_shape.maxInputShape);
    mprofile_->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kOPT, \
        dynamic_shape.optInputShape);
    mconfig_->addOptimizationProfile(mprofile_);  

    build_engine(engine_precision, optimize_level);
}

TRTEngine::TRTEngine(std::string onnx_path, EnginePrecision engine_precision, 
    Optimizationlevel optimize_level, std::vector<EngineDynamicShape> dynamic_shapes) {
    init_members(onnx_path);

    for (int input_id = 0; input_id < dynamic_shapes.size(); input_id++) {
        nvinfer1::ITensor* inputTensor = mnetwork_->getInput(input_id);
        mprofile_->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMIN, \
            dynamic_shapes[input_id].minInputShape);
        mprofile_->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMAX, \
            dynamic_shapes[input_id].maxInputShape);
        mprofile_->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kOPT, \
            dynamic_shapes[input_id].optInputShape);
        mconfig_->addOptimizationProfile(mprofile_);
    }
      
    build_engine(engine_precision, optimize_level);
}

void TRTEngine::init_members(std::string& onnx_path) {
    nvinfer1::IBuilder* mbuilder = mprovider_.get_trt_builder();
    TRTLogger mlogger = mprovider_.get_trt_logger();

    mconfig_ = mbuilder->createBuilderConfig();
    mnetwork_ = mbuilder->createNetworkV2(1U << 
        static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    mparser_ = nvonnxparser::createParser(*mnetwork_, mlogger);
    mprofile_ = mbuilder->createOptimizationProfile(); 

    // 2GB
    mconfig_->setMaxWorkspaceSize(1 << 30);
    std::vector<unsigned char> onnx_data = load_file(onnx_path);
    mparser_->parse(onnx_data.data(), onnx_data.size());
}

void TRTEngine::build_engine(EnginePrecision& engine_precision, Optimizationlevel& optimize_level) {
    mconfig_->setBuilderOptimizationLevel(static_cast<int>(optimize_level));

    switch (engine_precision) {
        case EnginePrecision::FP32:
            mconfig_->setFlag(nvinfer1::BuilderFlag::kTF32);
            break;
        case EnginePrecision::FP16:
            mconfig_->setFlag(nvinfer1::BuilderFlag::kFP16);
            break;
        case EnginePrecision::INT8:
            mconfig_->setFlag(nvinfer1::BuilderFlag::kFP16);
            mconfig_->setFlag(nvinfer1::BuilderFlag::kINT8);

            // 删除最后一个默认策略源，消除部分模型量化过程找不到策略源错误（例如yolov5-seg)
            mconfig_->setTacticSources(1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS)
                | 1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS_LT)
                | 1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUDNN)
                | 1U << static_cast<uint32_t>(nvinfer1::TacticSource::kEDGE_MASK_CONVOLUTIONS));

            // TODO Calibrator
            break;
        default:
            mconfig_->setFlag(nvinfer1::BuilderFlag::kTF32);
            break;
    }

    nvinfer1::IBuilder* mbuilder = mprovider_.get_trt_builder();
    mengine_ = mbuilder->buildEngineWithConfig(*mnetwork_, *mconfig_);
}

void TRTEngine::save_engine(std::string save_path) {
    if (mengine_) {
        mengine_data_ = mengine_->serialize();
        FILE* f = fopen(save_path.c_str(), "wb");
        fwrite(mengine_data_->data(), 1, mengine_data_->size(), f);
        fclose(f);
    }
}

TRTEngine::~TRTEngine() noexcept {
    if (mengine_data_) mengine_data_->destroy(), mengine_data_ = nullptr;
    if (mengine_) mengine_->destroy(), mengine_ = nullptr;
    if (mprofile_) mprofile_ = nullptr;
    if (mparser_) mparser_->destroy(), mparser_ = nullptr;
    if (mnetwork_) mnetwork_->destroy(), mnetwork_ = nullptr;
    if (mconfig_) mconfig_->destroy(), mconfig_ = nullptr;
}
}