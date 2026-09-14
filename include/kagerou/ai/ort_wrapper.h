// ============================================================================
// Kagerou AI Inference — ONNX Runtime C++ Wrapper
// ZERO-COPY GPU inference via CUDA EP (+ optional TensorRT EP)
// ============================================================================

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <cstdint>
#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>

namespace kagerou {
namespace ai {

// ============================================================================
// GpuTensor — owns device memory for model I/O
// ============================================================================
struct GpuTensor {
    uint8_t* data = nullptr;
    size_t   bytes = 0;

    void alloc(size_t sz);
    void free();
    bool ok() const { return data != nullptr; }
};

// ============================================================================
// TrtSession — wraps one ONNX model with CUDA/TensorRT execution
// ============================================================================
class TrtSession {
public:
    TrtSession() = default;
    ~TrtSession();
    TrtSession(const TrtSession&) = delete;
    TrtSession& operator=(const TrtSession&) = delete;

    bool load(const std::string& path, int device = 0, bool use_trt = true);
    bool run(std::vector<GpuTensor>& inputs, std::vector<GpuTensor>& outputs,
             const std::vector<std::vector<int64_t>>& input_shapes = {},
             std::vector<std::vector<int64_t>>* out_shapes = nullptr);
    void unload();

    size_t num_inputs() const;
    size_t num_outputs() const;
    std::vector<int64_t> input_shape(size_t i) const;
    std::vector<int64_t> output_shape(size_t i) const;
    std::string input_name(size_t i) const;
    std::string output_name(size_t i) const;
    bool loaded() const { return session_ != nullptr; }

private:
    std::unique_ptr<Ort::Env>        env_;
    std::unique_ptr<Ort::Session>    session_;
    std::unique_ptr<Ort::MemoryInfo> mem_;
    std::vector<std::string>         in_names_;
    std::vector<std::string>         out_names_;
};

// ============================================================================
// AiInference — singleton, manages sessions + GPU memory
// ============================================================================
class AiInference {
public:
    static AiInference& get();

    bool init(int device = 0);
    void shutdown();

    TrtSession* load(const std::string& tag, const std::string& model,
                      int device = 0, bool use_trt = true);
    TrtSession* find(const std::string& tag);

    GpuTensor gpu_alloc(size_t sz);
    void      gpu_free(GpuTensor& t);

    bool ready() const { return ready_; }

private:
    AiInference() = default;
    ~AiInference();
    bool ready_ = false;
    int  device_ = 0;
    std::unique_ptr<Ort::Env>        env_;
    std::unique_ptr<Ort::MemoryInfo> mem_;
    std::map<std::string, std::unique_ptr<TrtSession>> sessions_;
};

} // namespace ai
} // namespace kagerou
