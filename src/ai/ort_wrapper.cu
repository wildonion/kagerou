// ============================================================================
// Kagerou AI Inference — ONNX Runtime Wrapper Implementation
// GPU-only: CUDA EP + optional TensorRT EP. Zero CPU copies.
// ============================================================================

#include "kagerou/ai/ort_wrapper.h"
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#include <codecvt>
#include <locale>
static std::wstring to_wide(const std::string& s) {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
    return conv.from_bytes(s);
}
#define ORTSTR(s) to_wide(s).c_str()

static void ort_add_dll_dir() {
    // Add sdk/onnxruntime/lib and sdk/tensorrt to DLL search path
    wchar_t buf[MAX_PATH];
    if (GetModuleFileNameW(NULL, buf, MAX_PATH)) {
        std::wstring p(buf);
        auto pos = p.find_last_of(L"\\/");
        if (pos != std::wstring::npos) {
            std::wstring dir = p.substr(0, pos);
            SetDllDirectoryW((dir + L"\\..\\sdk\\onnxruntime\\lib").c_str());
            AddDllDirectory((dir + L"\\..\\sdk\\tensorrt").c_str());
        }
    }
}
#else
#define ORTSTR(s) (s).c_str()
static void ort_add_dll_dir() {}
#endif

namespace kagerou {
namespace ai {

// ============================================================================
// GpuTensor
// ============================================================================
void GpuTensor::alloc(size_t sz) {
    free();
    if (sz > 0 && cudaMalloc(&data, sz) == cudaSuccess) {
        bytes = sz;
        cudaMemset(data, 0, sz);
    }
}

void GpuTensor::free() {
    if (data) { cudaFree(data); data = nullptr; bytes = 0; }
}

// ============================================================================
// TrtSession
// ============================================================================
static size_t count_elements(const std::vector<int64_t>& s) {
    size_t n = 1; for (auto d : s) n *= (size_t)d; return n;
}

TrtSession::~TrtSession() { unload(); }

void TrtSession::unload() {
    session_.reset();
    mem_.reset();
    env_.reset();
    in_names_.clear();
    out_names_.clear();
}

// Absolute engine-cache dir next to the project (bin/../trt_engines).
// The old relative "trt_engines" resolved against the process working
// directory, so engines built from one CWD were never reused from another
// and every launch paid a full rebuild (plus its scary build log).
static std::string trt_cache_dir() {
#ifdef _WIN32
    char buf[MAX_PATH] = {};
    if (GetModuleFileNameA(NULL, buf, MAX_PATH)) {
        std::string p(buf);
        auto pos = p.find_last_of("\\/");
        if (pos != std::string::npos) {
            std::string d = p.substr(0, pos) + "\\..\\trt_engines";
            CreateDirectoryA(d.c_str(), NULL);
            DWORD attr = GetFileAttributesA(d.c_str());
            if (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY))
                return d;
        }
    }
#endif
    return "trt_engines";
}

bool TrtSession::load(const std::string& path, int device, bool use_trt) {
    unload();
    ort_add_dll_dir();
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Kagerou");

        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        so.SetIntraOpNumThreads(1);

        // TensorRT EP (primary) — fastest for conv-heavy models
        if (use_trt) {
            try {
                static std::string s_cache = trt_cache_dir();
                OrtTensorRTProviderOptions trt{};
                trt.device_id = device;
                trt.trt_fp16_enable = 1;
                trt.trt_engine_cache_enable = 1;
                trt.trt_engine_cache_path = s_cache.c_str();
                trt.trt_max_workspace_size = 1u * 1024u * 1024u * 1024u;
                trt.trt_max_partition_iterations = 10;
                trt.trt_min_subgraph_size = 1;
                so.AppendExecutionProvider_TensorRT(trt);
                fprintf(stderr, "[AI] TensorRT EP enabled (FP16, engine cache: %s)\n",
                        s_cache.c_str());
            } catch (const Ort::Exception& e) {
                fprintf(stderr, "[AI] TensorRT EP unavailable: %s\n", e.what());
                fprintf(stderr, "[AI] Falling back to CUDA EP only\n");
            }
        } else {
            fprintf(stderr, "[AI] TensorRT EP disabled (use_trt=false)\n");
        }

        // CUDA EP — fallback for ops TensorRT can't handle
        OrtCUDAProviderOptions cuda{};
        cuda.device_id = device;
        cuda.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
        cuda.gpu_mem_limit = SIZE_MAX;
        cuda.arena_extend_strategy = 0;
        cuda.do_copy_in_default_stream = 1;
        so.AppendExecutionProvider_CUDA(cuda);

        session_ = std::make_unique<Ort::Session>(*env_, ORTSTR(path), so);
        mem_ = std::make_unique<Ort::MemoryInfo>(
            "Cuda", OrtArenaAllocator, device, OrtMemTypeDefault);

        // Cache names
        Ort::AllocatorWithDefaultOptions a;
        for (size_t i = 0; i < session_->GetInputCount(); i++) {
            auto n = session_->GetInputNameAllocated(i, a);
            in_names_.push_back(n.get());
        }
        for (size_t i = 0; i < session_->GetOutputCount(); i++) {
            auto n = session_->GetOutputNameAllocated(i, a);
            out_names_.push_back(n.get());
        }

        fprintf(stderr, "[TRT] Loaded %s (%zu in, %zu out) on GPU %d\n",
                path.c_str(), in_names_.size(), out_names_.size(), device);
        return true;
    } catch (const Ort::Exception& e) {
        fprintf(stderr, "[TRT] Load failed %s: %s\n", path.c_str(), e.what());
        unload();
        return false;
    }
}

size_t TrtSession::num_inputs()  const { return in_names_.size(); }
size_t TrtSession::num_outputs() const { return out_names_.size(); }

std::vector<int64_t> TrtSession::input_shape(size_t i) const {
    if (!session_ || i >= session_->GetInputCount()) return {};
    auto t = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
    return t.GetShape();
}

std::vector<int64_t> TrtSession::output_shape(size_t i) const {
    if (!session_ || i >= session_->GetOutputCount()) return {};
    auto t = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
    return t.GetShape();
}

std::string TrtSession::input_name(size_t i) const {
    return i < in_names_.size() ? in_names_[i] : "";
}

std::string TrtSession::output_name(size_t i) const {
    return i < out_names_.size() ? out_names_[i] : "";
}

bool TrtSession::run(std::vector<GpuTensor>& inputs,
                     std::vector<GpuTensor>& outputs,
                     const std::vector<std::vector<int64_t>>& input_shapes,
                     std::vector<std::vector<int64_t>>* out_shapes) {
    if (!session_) return false;

    try {
        Ort::IoBinding b(*session_);

        // Bind inputs — GPU pointers directly, no copy
        for (size_t i = 0; i < in_names_.size(); i++) {
            std::vector<int64_t> sh;
            if (i < input_shapes.size() && !input_shapes[i].empty()) {
                sh = input_shapes[i];
            } else {
                sh = input_shape(i);
                size_t buf_floats = inputs[i].bytes / sizeof(float);

                // Resolve dynamic dims: batch=1, rest computed from buffer size
                size_t known_spatial = 1;
                int neg_count = 0;
                for (size_t d = 0; d < sh.size(); d++) {
                    if (d == 0 && sh[d] <= 0) { sh[d] = 1; continue; }
                    if (sh[d] > 0) known_spatial *= (size_t)sh[d];
                    else neg_count++;
                }
                if (neg_count > 0 && known_spatial > 0) {
                    size_t spatial_floats = buf_floats / (size_t)sh[0];
                    if (neg_count == 1) {
                        for (size_t d = 1; d < sh.size(); d++) {
                            if (sh[d] <= 0) {
                                sh[d] = (int64_t)(spatial_floats / known_spatial);
                                if (sh[d] <= 0) sh[d] = 1;
                            }
                        }
                    } else {
                        size_t spatial_area = spatial_floats / known_spatial;
                        size_t side = 1;
                        while ((side + 1) * (side + 1) <= spatial_area) side++;
                        bool first = true;
                        for (size_t d = 1; d < sh.size(); d++) {
                            if (sh[d] <= 0) {
                                if (first) { sh[d] = (int64_t)side; first = false; }
                                else { sh[d] = (int64_t)(spatial_area / side); if (sh[d]<=0) sh[d]=1; }
                            }
                        }
                    }
                }
            }
            b.BindInput(in_names_[i].c_str(),
                Ort::Value::CreateTensor(*mem_, inputs[i].data, inputs[i].bytes,
                    sh.data(), sh.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        }

        // Allocate + bind outputs
        // Let ORT infer the actual output shapes by not specifying them
        // ORT will allocate the correct GPU memory for outputs
        outputs.resize(out_names_.size());
        for (size_t i = 0; i < out_names_.size(); i++) {
            // Don't pre-allocate — let ORT figure out the shape
            // ORT's CUDA EP will allocate GPU memory internally
            b.BindOutput(out_names_[i].c_str(), *mem_);
        }

        session_->Run(Ort::RunOptions{}, b);

        // Extract ORT-allocated output GPU memory into our GpuTensors
        auto output_names_out = b.GetOutputValues();
        if (out_shapes) {
            out_shapes->clear();
            for (auto& val : output_names_out)
                out_shapes->push_back(val.GetTensorTypeAndShapeInfo().GetShape());
        }
        for (size_t i = 0; i < output_names_out.size() && i < outputs.size(); i++) {
            auto& val = output_names_out[i];
            auto shape = val.GetTensorTypeAndShapeInfo().GetShape();
            auto* ptr = val.GetTensorMutableData<void>();
            size_t sz = count_elements(shape) * sizeof(float);
            outputs[i].alloc(sz);
            cudaMemcpyAsync(outputs[i].data, ptr, sz, cudaMemcpyDeviceToDevice, 0);
        }
        cudaDeviceSynchronize();
        return true;
    } catch (const Ort::Exception& e) {
        fprintf(stderr, "[TRT] Run failed: %s\n", e.what());
        return false;
    }
}

// ============================================================================
// AiInference
// ============================================================================
AiInference& AiInference::get() {
    static AiInference inst;
    return inst;
}

AiInference::~AiInference() { shutdown(); }

bool AiInference::init(int device) {
    if (ready_) return true;
    ort_add_dll_dir();
    try {
        device_ = device;
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Kagerou");
        mem_ = std::make_unique<Ort::MemoryInfo>(
            "Cuda", OrtArenaAllocator, device, OrtMemTypeDefault);
        ready_ = true;
        fprintf(stderr, "[AI] Inference engine ready (GPU %d)\n", device);
        return true;
    } catch (...) {
        return false;
    }
}

void AiInference::shutdown() {
    sessions_.clear();
    mem_.reset();
    env_.reset();
    ready_ = false;
}

TrtSession* AiInference::load(const std::string& tag,
                              const std::string& model,
                              int device, bool use_trt) {
    if (!ready_) return nullptr;
    auto it = sessions_.find(tag);
    if (it != sessions_.end() && it->second->loaded()) return it->second.get();

    auto s = std::make_unique<TrtSession>();
    if (!s->load(model, device < 0 ? device_ : device, use_trt)) return nullptr;
    auto* p = s.get();
    sessions_[tag] = std::move(s);
    return p;
}

TrtSession* AiInference::find(const std::string& tag) {
    auto it = sessions_.find(tag);
    return (it != sessions_.end()) ? it->second.get() : nullptr;
}

GpuTensor AiInference::gpu_alloc(size_t sz) {
    GpuTensor t; t.alloc(sz); return t;
}

void AiInference::gpu_free(GpuTensor& t) { t.free(); }

} // namespace ai
} // namespace kagerou
