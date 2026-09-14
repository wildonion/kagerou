// ============================================================================
// Kagerou AI Object Detection — YOLOv8n (Ultralytics, 640x640, COCO 80)
// GPU preprocess → ORT inference → CPU decode + NMS → DetectBox list.
// Export convention (verified on-device): static input images[1,3,640,640]
// RGB/255, output output0[1,84,8400] = cx,cy,w,h (0..640 px) + 80 class
// scores WITH sigmoid already applied in-graph (Sigmoid→Concat tail).
// ============================================================================

#include "kagerou/ai/ai_filters.h"
#include "kagerou/ai/ort_wrapper.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

namespace kagerou {
namespace ai {

void launch_letterbox_resize_rgb(const uint8_t* src, uint8_t* dst,
                                 uint32_t src_w, uint32_t src_h,
                                 uint32_t dst_w, uint32_t dst_h,
                                 float& out_scale, float& out_pad_x, float& out_pad_y,
                                 cudaStream_t s);

#define YOLO_DIM 640
#define YOLO_CELLS 8400
#define YOLO_CLASSES 80
#define YOLO_CONF_THRESH 0.35f
#define YOLO_NMS_IOU 0.45f

// RGB [0,255] -> NCHW float [0,1] (no channel swap: Ultralytics trains RGB).
__global__ void yolo_pack_kernel(const uint8_t* __restrict__ rgb,
                                 float* __restrict__ nchw,
                                 uint32_t w, uint32_t h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    uint32_t plane = w * h, pixel = y * w + x, si = pixel * 3;
    nchw[0 * plane + pixel] = rgb[si + 0] * (1.0f / 255.0f);
    nchw[1 * plane + pixel] = rgb[si + 1] * (1.0f / 255.0f);
    nchw[2 * plane + pixel] = rgb[si + 2] * (1.0f / 255.0f);
}

static bool g_yolo_init = false;

bool ai_detect_init(const char* model_path, int device_id) {
    if (g_yolo_init) return true;
    auto& eng = AiInference::get();
    if (!eng.init(device_id)) return false;
    auto* s = eng.load("yolo", model_path, device_id);
    if (!s) { fprintf(stderr, "[AI Detect] Cannot load %s\n", model_path); return false; }
    fprintf(stderr, "[AI Detect] Model ready.\n");
    g_yolo_init = true;
    return true;
}

struct YoloDet { float x1, y1, x2, y2, score; int cls; };

static float yolo_iou(const YoloDet& a, const YoloDet& b) {
    float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float iw = ix2 - ix1, ih = iy2 - iy1;
    if (iw <= 0 || ih <= 0) return 0;
    float inter = iw * ih;
    float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return ua > 0 ? inter / ua : 0;
}

int ai_detect(const uint8_t* d_src, uint32_t w, uint32_t h,
              DetectBox* boxes_out, int max_boxes, cudaStream_t stream) {
    if (!g_yolo_init || max_boxes <= 0) return 0;
    auto& eng = AiInference::get();
    auto* s = eng.find("yolo");
    if (!s) return 0;

    // 1. letterbox to 640x640 (aspect preserved — stretching breaks boxes)
    GpuTensor d_rs, d_in;
    d_rs.alloc(YOLO_DIM * YOLO_DIM * 3);
    d_in.alloc(3 * YOLO_DIM * YOLO_DIM * sizeof(float));
    if (!d_rs.ok() || !d_in.ok()) return 0;
    float lb_scale = 1, lb_px = 0, lb_py = 0;
    launch_letterbox_resize_rgb(d_src, d_rs.data, w, h,
                                YOLO_DIM, YOLO_DIM,
                                lb_scale, lb_px, lb_py, stream);
    {
        dim3 b(16, 16), g((YOLO_DIM + 15) / 16, (YOLO_DIM + 15) / 16);
        yolo_pack_kernel<<<g, b, 0, stream>>>(
            d_rs.data, reinterpret_cast<float*>(d_in.data), YOLO_DIM, YOLO_DIM);
    }
    cudaStreamSynchronize(stream);
    eng.gpu_free(d_rs);

    // 2. run (single input)
    std::vector<int64_t> shape = {1, 3, YOLO_DIM, YOLO_DIM};
    std::vector<std::vector<int64_t>> shapes = {shape};
    std::vector<GpuTensor> inputs = {d_in};
    std::vector<GpuTensor> outputs;
    if (!s->run(inputs, outputs, shapes)) {
        eng.gpu_free(d_in);
        for (auto& o : outputs) eng.gpu_free(o);
        return 0;
    }
    eng.gpu_free(d_in);
    if (outputs.empty()) {
        for (auto& o : outputs) eng.gpu_free(o);
        return 0;
    }

    // 3. download [84][8400]
    std::vector<float> out(84 * YOLO_CELLS);
    cudaMemcpyAsync(out.data(), outputs[0].data, out.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for (auto& o : outputs) eng.gpu_free(o);

    // 4. decode: best-class per cell, conf threshold, back to frame dims
    // (undo the letterbox: unpad, then unscale)
    std::vector<YoloDet> dets;
    dets.reserve(128);
    for (int i = 0; i < YOLO_CELLS; i++) {
        float best = 0;
        int bcls = -1;
        for (int c = 0; c < YOLO_CLASSES; c++) {
            float v = out[(4 + c) * YOLO_CELLS + i];
            if (v > best) { best = v; bcls = c; }
        }
        if (best < YOLO_CONF_THRESH || bcls < 0) continue;
        float cx = out[0 * YOLO_CELLS + i], cy = out[1 * YOLO_CELLS + i];
        float bw = out[2 * YOLO_CELLS + i], bh = out[3 * YOLO_CELLS + i];
        YoloDet d;
        d.x1 = (cx - bw / 2 - lb_px) / lb_scale;
        d.y1 = (cy - bh / 2 - lb_py) / lb_scale;
        d.x2 = (cx + bw / 2 - lb_px) / lb_scale;
        d.y2 = (cy + bh / 2 - lb_py) / lb_scale;
        if (d.x1 < 0) d.x1 = 0; if (d.y1 < 0) d.y1 = 0;
        if (d.x2 > (float)w) d.x2 = (float)w;
        if (d.y2 > (float)h) d.y2 = (float)h;
        d.score = best; d.cls = bcls;
        if (d.x2 > d.x1 && d.y2 > d.y1) dets.push_back(d);
    }

    // 5. NMS (class-agnostic, like ultralytics default)
    std::sort(dets.begin(), dets.end(),
              [](const YoloDet& a, const YoloDet& b) { return a.score > b.score; });
    std::vector<YoloDet> kept;
    for (auto& d : dets) {
        bool drop = false;
        for (auto& k : kept) {
            if (yolo_iou(d, k) > YOLO_NMS_IOU) { drop = true; break; }
        }
        if (!drop) {
            kept.push_back(d);
            if ((int)kept.size() >= max_boxes) break;
        }
    }
    for (size_t i = 0; i < kept.size(); i++) {
        boxes_out[i].x1 = kept[i].x1; boxes_out[i].y1 = kept[i].y1;
        boxes_out[i].x2 = kept[i].x2; boxes_out[i].y2 = kept[i].y2;
        boxes_out[i].score = kept[i].score; boxes_out[i].cls = kept[i].cls;
    }
    // Clear the tail so stale boxes from a busier frame can't ghost.
    for (size_t i = kept.size(); i < (size_t)max_boxes; i++) {
        boxes_out[i].x1 = boxes_out[i].y1 = boxes_out[i].x2 = boxes_out[i].y2 = 0;
        boxes_out[i].score = 0; boxes_out[i].cls = -1;
    }
    return (int)kept.size();
}

// COCO 80 (must match models/detect/coco80.txt order)
static const char* kCocoNames[YOLO_CLASSES] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
};

const char* ai_detect_class_name(int cls) {
    if (cls < 0 || cls >= YOLO_CLASSES) return "?";
    return kCocoNames[cls];
}

void ai_detect_color(int cls, uint8_t* rgb_out3) {
    if (cls < 0) cls = 0;
    rgb_out3[0] = (uint8_t)(80 + (cls * 149) % 176);
    rgb_out3[1] = (uint8_t)(80 + (cls * 97) % 176);
    rgb_out3[2] = (uint8_t)(80 + (cls * 61) % 176);
}

} // namespace ai
} // namespace kagerou
