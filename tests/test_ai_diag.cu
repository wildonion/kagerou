// Diagnostic: probe AI model I/O domains with a REAL camera frame.
// Usage: bin\test_ai_diag.exe <rgb24_frame> [frames_in_file]
// Reads frame #0 (640x480 rgb24), runs pose + denoise (both scalings),
// prints output stats. No kernels, plain memcpys.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>

#include "kagerou/ai/ort_wrapper.h"
#include "kagerou/ai/ai_filters.h"
#include "../src/ai/ort_wrapper.cu"
#include "../src/ai/ai_preprocess.cu"
#include "../src/ai/ai_denoise.cu"
#include "../src/ai/ai_face.cu"
#include "../src/ai/ai_yolo.cu"
#include "../src/filters/creative.cu"
#include "../src/ai/ai_depth.cu"
#include "../src/ai/ai_matting.cu"

using namespace kagerou::ai;

static void stats_ch(const float* d, int w, int h, const char* tag) {
    size_t n = (size_t)w * h;
    std::vector<float> hbuf(n * 3);
    cudaMemcpy(hbuf.data(), d, n * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int c = 0; c < 3; c++) {
        float mn = 1e30f, mx = -1e30f, sum = 0;
        for (size_t i = 0; i < n; i++) {
            float v = hbuf[i * 3 + c];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
        }
        printf("  %s ch%d: min=%+.4f max=%+.4f mean=%+.4f\n", tag, c, mn, mx, sum / n);
    }
}

int main(int argc, char** argv) {
    const char* frame_path = (argc > 1) ? argv[1] : "camframe.raw";
    FILE* f = fopen(frame_path, "rb");
    if (!f) { printf("cannot open %s\n", frame_path); return 1; }
    int W = 640, H = 480;
    if (argc > 2) W = atoi(argv[2]);
    if (argc > 3) H = atoi(argv[3]);
    std::vector<uint8_t> rgb((size_t)W * H * 3);
    if (fread(rgb.data(), 1, rgb.size(), f) != rgb.size()) { printf("short read\n"); return 1; }
    fclose(f);
    // input stats
    double sum[3] = {};
    for (size_t i = 0; i < (size_t)W * H; i++)
        for (int c = 0; c < 3; c++) sum[c] += rgb[i * 3 + c];
    printf("input rgb mean: R=%.1f G=%.1f B=%.1f\n", sum[0] / (W*H), sum[1] / (W*H), sum[2] / (W*H));

    if (!AiInference::get().init(0)) { printf("ORT init failed\n"); return 1; }

    // ---------- POSE ----------
    {
        auto* s = AiInference::get().load("pose", "models/pose/pose_landmark.onnx", 0);
        if (!s) { printf("pose load FAILED\n"); }
        else {
            // CPU nearest resize to 256 + NHWC float [0,1]
            const int D = 256;
            std::vector<float> in((size_t)D * D * 3);
            for (int y = 0; y < D; y++) for (int x = 0; x < D; x++) {
                int sx = x * W / D, sy = y * H / D;
                for (int c = 0; c < 3; c++)
                    in[((size_t)y * D + x) * 3 + c] = rgb[((size_t)sy * W + sx) * 3 + c] / 255.0f;
            }
            GpuTensor d_in; d_in.alloc(in.size() * sizeof(float));
            cudaMemcpy(d_in.data, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice);
            std::vector<GpuTensor> inputs = {d_in}, outputs;
            std::vector<std::vector<int64_t>> shapes = {{1, 256, 256, 3}};
            bool ok = s->run(inputs, outputs, shapes);
            printf("pose run: %s\n", ok ? "OK" : "FAIL");
            if (ok && !outputs.empty()) {
                std::vector<float> lm(195);
                cudaMemcpy(lm.data(), outputs[0].data, 195 * sizeof(float), cudaMemcpyDeviceToHost);
                int vis = 0;
                for (int i = 0; i < 33; i++) {
                    float x = lm[i*5+0], y = lm[i*5+1], v = lm[i*5+3], p = lm[i*5+4];
                    if (i < 5) printf("  lm%d: x=%+.3f y=%+.3f z=%+.3f vis=%.3f pres=%.3f\n", i, x, y, lm[i*5+2], v, p);
                    if (v > 0.5f) vis++;
                }
                printf("  landmarks with vis>0.5: %d/33\n", vis);
            }
            AiInference::get().gpu_free(d_in);
            for (auto& o : outputs) AiInference::get().gpu_free(o);
        }
    }

    // ---------- DENOISE full app path (5 identical frames, real noise map) ----------
    {
        bool init_ok = ai_denoise_init("models/denoise/fastdvdnet.onnx", 0);
        printf("ai_denoise_init: %s\n", init_ok ? "OK" : "FAIL");
        if (init_ok) {
            uint8_t* fr[5];
            for (int i = 0; i < 5; i++) {
                cudaMalloc(&fr[i], rgb.size());
                cudaMemcpy(fr[i], rgb.data(), rgb.size(), cudaMemcpyHostToDevice);
            }
            uint8_t* d_out = nullptr;
            cudaMalloc(&d_out, rgb.size());
            cudaStream_t st = 0;
            bool ok = ai_denoise(fr[0], fr[1], fr[2], fr[3], fr[4], d_out, W, H, st);
            printf("ai_denoise() app path: %s\n", ok ? "OK" : "FAIL");
            if (ok) {
                std::vector<uint8_t> out(rgb.size());
                cudaMemcpy(out.data(), d_out, rgb.size(), cudaMemcpyDeviceToHost);
                double m[3] = {};
                for (size_t i = 0; i < (size_t)W * H; i++)
                    for (int c = 0; c < 3; c++) m[c] += out[i * 3 + c];
                printf("  out rgb mean: R=%.1f G=%.1f B=%.1f (in was 189.2/188.1/188.3)\n",
                       m[0] / (W*H), m[1] / (W*H), m[2] / (W*H));
            }
            for (int i = 0; i < 5; i++) cudaFree(fr[i]);
            cudaFree(d_out);
        }
    }

    // ---------- FACE variants (preprocess x decode matrix on real face) ----------
    static kagerou::ai::FaceBox g_face_probe[8];
    static int g_nface_probe = 0;
    {
        auto* s = AiInference::get().load("face", "models/face/yunet_2023mar.onnx", 0);
        if (!s) { printf("face load FAILED\n"); }
        else {
            const int D = 640;
            const int CNT[3] = {6400, 1600, 400};
            const int ST[3] = {8, 16, 32};
            for (int v = 0; v < 4; v++) {
                // v: bit0 = raw[0,255] input (else [0,1]); bit1 = RGB order (else BGR)
                bool raw = (v & 1) != 0, rgbord = (v & 2) != 0;
                std::vector<float> in(3 * D * D);
                for (int y = 0; y < D; y++) for (int x = 0; x < D; x++) {
                    int sx = x * W / D, sy = y * H / D;
                    float r = rgb[((size_t)sy * W + sx) * 3 + 0];
                    float g = rgb[((size_t)sy * W + sx) * 3 + 1];
                    float b = rgb[((size_t)sy * W + sx) * 3 + 2];
                    float sc = raw ? 1.0f : 1.0f / 255.0f;
                    if (rgbord) { in[(0*D+y)*D+x] = r*sc; in[(1*D+y)*D+x] = g*sc; in[(2*D+y)*D+x] = b*sc; }
                    else { in[(0*D+y)*D+x] = b*sc; in[(1*D+y)*D+x] = g*sc; in[(2*D+y)*D+x] = r*sc; }
                }
                GpuTensor d_in; d_in.alloc(in.size() * sizeof(float));
                cudaMemcpy(d_in.data, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice);
                std::vector<GpuTensor> inputs = {d_in}, outputs;
                std::vector<std::vector<int64_t>> shapes = {{1, 3, D, D}};
                bool ok = s->run(inputs, outputs, shapes);
                int ndet = 0;
                float best = 0;
                if (ok && outputs.size() >= 9) {
                    std::vector<float> cls[3], obj[3], box[3];
                    for (int l = 0; l < 3; l++) {
                        cls[l].resize(CNT[l]); obj[l].resize(CNT[l]); box[l].resize(CNT[l] * 4);
                        cudaMemcpy(cls[l].data(), outputs[l].data, CNT[l] * sizeof(float), cudaMemcpyDeviceToHost);
                        cudaMemcpy(obj[l].data(), outputs[3+l].data, CNT[l] * sizeof(float), cudaMemcpyDeviceToHost);
                        cudaMemcpy(box[l].data(), outputs[6+l].data, CNT[l] * 4 * sizeof(float), cudaMemcpyDeviceToHost);
                    }
                    const int ST[3] = {8, 16, 32};
                    for (int l = 0; l < 3; l++) {
                        int g = D / ST[l];
                        for (int i = 0; i < CNT[l]; i++) {
                            float sc2 = 1.0f / (1.0f + expf(-cls[l][i])) * 1.0f / (1.0f + expf(-obj[l][i]));
                            if (sc2 > best) best = sc2;
                            if (sc2 > 0.5f) {
                                ndet++;
                                if (ndet <= 4) {
                                    int yy = i / g, xx = i % g;
                                    float px = (xx + 0.5f) * ST[l], py = (yy + 0.5f) * ST[l];
                                    float cx = px + box[l][i*4+0] * ST[l];
                                    float cy = py + box[l][i*4+1] * ST[l];
                                    float bw = expf(box[l][i*4+2]) * ST[l];
                                    float bh = expf(box[l][i*4+3]) * ST[l];
                                    printf("    det l=%d k=%d s=%.3f box640=[%.0f,%.0f,%.0f,%.0f]\n",
                                           l, i, sc2, cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2);
                                }
                            }
                        }
                    }
                }
                printf("face variant %d (%s %s): run=%s dets>0.5=%d best=%.3f\n",
                       v, raw ? "raw255" : "[0,1]", rgbord ? "RGB" : "BGR",
                       ok ? "OK" : "FAIL", ndet, best);
                AiInference::get().gpu_free(d_in);
                for (auto& o : outputs) AiInference::get().gpu_free(o);
            }
        }
    }

    // ---------- PALM (dual convention: [0,1] vs [-1,1]) ----------
    {
        auto* s = AiInference::get().load("palm", "models/hands/palm_detection_lite.onnx", 0);
        if (!s) { printf("palm load FAILED\n"); }
        else {
            const int D = 192;
            for (int trial = 0; trial < 2; trial++) {
                std::vector<float> in(3 * D * D);
                for (int y = 0; y < D; y++) for (int x = 0; x < D; x++) {
                    int sx = x * W / D, sy = y * H / D;
                    for (int c = 0; c < 3; c++) {
                        float v = rgb[((size_t)sy * W + sx) * 3 + c] / 255.0f;
                        in[(c * D + y) * D + x] = (trial == 0) ? v : v * 2.0f - 1.0f;
                    }
                }
                GpuTensor d_in; d_in.alloc(in.size() * sizeof(float));
                cudaMemcpy(d_in.data, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice);
                std::vector<GpuTensor> inputs = {d_in}, outputs;
                std::vector<std::vector<int64_t>> shapes = {{1, 3, D, D}};
                bool ok = s->run(inputs, outputs, shapes);
                printf("palm run [%s]: %s\n", trial == 0 ? "0,1" : "-1,1", ok ? "OK" : "FAIL");
                if (ok && outputs.size() >= 2) {
                    std::vector<float> o(2016 * 18);
                    cudaMemcpy(o.data(), outputs[0].data, o.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::vector<float> sc(2016);
                    cudaMemcpy(sc.data(), outputs[1].data, sc.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    int npos = 0; float mx = -1e30f; int bi = 0;
                    for (int i = 0; i < 2016; i++) {
                        float s = 1.0f / (1.0f + expf(-sc[i]));
                        if (s > mx) { mx = s; bi = i; }
                        if (sc[i] > 0) npos++;
                    }
                    printf("  score: npos=%d best=%.4f at anchor %d\n", npos, mx, bi);
                    printf("  best anchor box+: ");
                    for (int c = 0; c < 18; c++) printf("%+.2f ", o[bi * 18 + c]);
                    printf("\n");
                }
                AiInference::get().gpu_free(d_in);
                for (auto& o : outputs) AiInference::get().gpu_free(o);
            }
        }
    }

    // ---------- FACE end-to-end (fills box for iris) ----------
    {
        printf("face e2e init: %s\n", ai_face_init("models/face/yunet_2023mar.onnx", 0) ? "OK" : "FAIL");
        uint8_t* d_f = nullptr;
        cudaMalloc(&d_f, rgb.size());
        cudaMemcpy(d_f, rgb.data(), rgb.size(), cudaMemcpyHostToDevice);
        g_nface_probe = ai_face(d_f, W, H, g_face_probe, 8, 0);
        printf("face e2e found: %d\n", g_nface_probe);
        for (int i = 0; i < g_nface_probe; i++)
            printf("  e2e face%d: [%.0f,%.0f,%.0f,%.0f] score=%.3f\n", i,
                   g_face_probe[i].x1, g_face_probe[i].y1,
                   g_face_probe[i].x2, g_face_probe[i].y2, g_face_probe[i].score);
        cudaFree(d_f);
    }

    // ---------- DETECT end-to-end (YOLOv8n on the real frame) ----------
    {
        printf("detect e2e init: %s\n", ai_detect_init("models/detect/yolov8n.onnx", 0) ? "OK" : "FAIL");
        uint8_t* d_f = nullptr;
        cudaMalloc(&d_f, rgb.size());
        cudaMemcpy(d_f, rgb.data(), rgb.size(), cudaMemcpyHostToDevice);
        static kagerou::ai::DetectBox det_probe[16];
        int ndet = ai_detect(d_f, W, H, det_probe, 16, 0);
        printf("detect e2e found: %d\n", ndet);
        for (int i = 0; i < ndet; i++)
            printf("  e2e det%d: %s [%.0f,%.0f,%.0f,%.0f] score=%.3f\n", i,
                   ai_detect_class_name(det_probe[i].cls),
                   det_probe[i].x1, det_probe[i].y1,
                   det_probe[i].x2, det_probe[i].y2, det_probe[i].score);
        cudaFree(d_f);
    }

    // ---------- IRIS (64x64 eye crop from detected face, both norms) ----------
    {
        auto* s = AiInference::get().load("iris", "models/gaze/iris_landmark.onnx", 0);
        if (!s) { printf("iris load FAILED\n"); }
        else if (g_nface_probe == 0) { printf("iris skipped: no face box\n"); }
        else {
            // left-eye crop from face box 0 (eyes ~40% height, 30%/70% width)
            const int D = 64;
            float fx1 = g_face_probe[0].x1, fy1 = g_face_probe[0].y1;
            float fw = g_face_probe[0].x2 - fx1, fh = g_face_probe[0].y2 - fy1;
            int ex = (int)(fx1 + 0.3f * fw) - D / 2, ey = (int)(fy1 + 0.4f * fh) - D / 2;
            if (ex < 0) ex = 0; if (ey < 0) ey = 0;
            if (ex + D > W) ex = W - D; if (ey + D > H) ey = H - D;
            printf("iris eyecrop at (%d,%d)\n", ex, ey);
            for (int trial = 0; trial < 2; trial++) {
                std::vector<float> in(3 * D * D);
                for (int y = 0; y < D; y++) for (int x = 0; x < D; x++) {
                    for (int c = 0; c < 3; c++) {
                        float v = rgb[((size_t)(ey + y) * W + ex + x) * 3 + c] / 255.0f;
                        in[(c * D + y) * D + x] = (trial == 0) ? v * 2.0f - 1.0f : v;
                    }
                }
                GpuTensor d_in; d_in.alloc(in.size() * sizeof(float));
                cudaMemcpy(d_in.data, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice);
                std::vector<GpuTensor> inputs = {d_in}, outputs;
                std::vector<std::vector<int64_t>> shapes = {{1, 3, D, D}};
                bool ok = s->run(inputs, outputs, shapes);
                printf("iris run [%s]: %s\n", trial == 0 ? "-1,1" : "0,1", ok ? "OK" : "FAIL");
                if (ok && outputs.size() >= 2) {
                    std::vector<float> iris(15), cont(213);
                    cudaMemcpy(iris.data(), outputs[1].data, 15 * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(cont.data(), outputs[0].data, 213 * sizeof(float), cudaMemcpyDeviceToHost);
                    printf("  iris pts:");
                    for (int i = 0; i < 5; i++) printf(" (%+.1f,%+.1f,%+.1f)", iris[i*3], iris[i*3+1], iris[i*3+2]);
                    printf("\n");
                }
                AiInference::get().gpu_free(d_in);
                for (auto& o : outputs) AiInference::get().gpu_free(o);
            }
        }
    }

    // ---------- MATTING full path on real frame ----------
    {
        printf("matting init: %s\n", ai_matting_init("models/matting/rvm_mobilenetv3_static.onnx", 0) ? "OK" : "FAIL");
        uint8_t* d_f = nullptr;
        cudaMalloc(&d_f, rgb.size());
        cudaMemcpy(d_f, rgb.data(), rgb.size(), cudaMemcpyHostToDevice);
        uint8_t* d_o = nullptr;
        cudaMalloc(&d_o, rgb.size());
        // run twice (2nd uses warmed engine); poi: do we get a person blob?
        for (int t = 0; t < 2; t++) {
            bool ok = ai_matting(d_f, d_o, W, H, 0);
            printf("matting run %d: %s\n", t, ok ? "OK" : "FAIL");
        }
        std::vector<uint8_t> out(rgb.size());
        cudaMemcpy(out.data(), d_o, rgb.size(), cudaMemcpyDeviceToHost);
        double m[3] = {};
        for (size_t i = 0; i < (size_t)W * H; i++)
            for (int c = 0; c < 3; c++) m[c] += out[i * 3 + c];
        printf("  composite mean: R=%.1f G=%.1f B=%.1f (in 153.1/127.7/111.0)\n",
               m[0] / (W*H), m[1] / (W*H), m[2] / (W*H));
        cudaFree(d_f);
        cudaFree(d_o);
    }

    // ---------- MATTING alpha mass on real frame ----------
    {
        auto* s = AiInference::get().find("matting");
        if (s) {
            const int MW = W, MH = H;
            std::vector<float> in(3 * MW * MH);
            for (int y = 0; y < MH; y++) for (int x = 0; x < MW; x++) {
                for (int c = 0; c < 3; c++)
                    in[(c * MH + y) * MW + x] = rgb[((size_t)y * W + x) * 3 + c] / 255.0f;
            }
            GpuTensor d_in; d_in.alloc(in.size() * sizeof(float));
            cudaMemcpy(d_in.data, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice);
            GpuTensor d_r[4], d_ds;
            int ch[] = {16, 20, 40, 64};
            for (int i = 0; i < 4; i++) d_r[i].alloc(ch[i] * sizeof(float));
            d_ds.alloc(sizeof(float));
            float ds = 0.5f;
            cudaMemcpy(d_ds.data, &ds, sizeof(float), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            std::vector<GpuTensor> inputs = {d_in, d_r[0], d_r[1], d_r[2], d_r[3], d_ds};
            std::vector<std::vector<int64_t>> shapes = {{1,3,MH,MW},{1,16,1,1},{1,20,1,1},{1,40,1,1},{1,64,1,1},{1}};
            std::vector<GpuTensor> outputs;
            bool ok = s->run(inputs, outputs, shapes);
            printf("matting alpha run: %s\n", ok ? "OK" : "FAIL");
            if (ok && outputs.size() >= 2) {
                std::vector<float> pha(MW * MH);
                cudaMemcpy(pha.data(), outputs[1].data, pha.size() * sizeof(float), cudaMemcpyDeviceToHost);
                int hi = 0, x0 = MW, y0 = MH, x1 = 0, y1 = 0;
                for (int y = 0; y < MH; y++) for (int x = 0; x < MW; x++) {
                    if (pha[y * MW + x] > 0.5f) {
                        hi++;
                        if (x < x0) x0 = x; if (x > x1) x1 = x;
                        if (y < y0) y0 = y; if (y > y1) y1 = y;
                    }
                }
                printf("  alpha>0.5: %.1f%% bbox=[%d,%d,%d,%d] (working %dx%d)\n",
                       100.0 * hi / (MW * MH), x0, y0, x1, y1, MW, MH);
                // cross-check vs YuNet face box: person region must be brighter
                if (g_nface_probe > 0) {
                    float fx1 = g_face_probe[0].x1 * MW / W, fy1 = g_face_probe[0].y1 * MH / H;
                    float fx2 = g_face_probe[0].x2 * MW / W, fy2 = g_face_probe[0].y2 * MH / H;
                    double si = 0, so = 0; int ni = 0, no = 0;
                    for (int y = 0; y < MH; y++) for (int x = 0; x < MW; x++) {
                        if (x >= fx1 && x < fx2 && y >= fy1 && y < fy2) { si += pha[y * MW + x]; ni++; }
                        else { so += pha[y * MW + x]; no++; }
                    }
                    printf("  alpha mean inside face box=%.3f outside=%.3f\n", si / ni, so / no);
                }
            }
            AiInference::get().gpu_free(d_in);
            for (int i = 0; i < 4; i++) AiInference::get().gpu_free(d_r[i]);
            AiInference::get().gpu_free(d_ds);
            for (auto& o : outputs) AiInference::get().gpu_free(o);
        }
    }

    // ---------- DENOISE (two scalings) ----------
    {
        auto* s = AiInference::get().load("fastdvdnet", "models/denoise/fastdvdnet.onnx", 0);
        if (!s) { printf("denoise load FAILED\n"); }
        else {
            for (int trial = 0; trial < 2; trial++) {
                float lo = (trial == 0) ? 0.0f : -1.0f; // input range under test
                size_t plane = (size_t)W * H;
                std::vector<float> in(15 * plane), nz(plane);
                for (size_t i = 0; i < plane; i++) {
                    for (int c = 0; c < 3; c++)
                        for (int fr = 0; fr < 5; fr++)
                            in[(fr * 3 + c) * plane + i] = lo + (rgb[i * 3 + c] / 255.0f) * (trial == 0 ? 1.0f : 2.0f);
                    nz[i] = 0.05f;
                }
                GpuTensor d_b, d_n;
                d_b.alloc(in.size() * sizeof(float));
                d_n.alloc(nz.size() * sizeof(float));
                cudaMemcpy(d_b.data, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_n.data, nz.data(), nz.size() * sizeof(float), cudaMemcpyHostToDevice);
                std::vector<GpuTensor> inputs = {d_b, d_n}, outputs;
                std::vector<std::vector<int64_t>> shapes = {{1,15,H,W},{1,1,H,W}};
                bool ok = s->run(inputs, outputs, shapes);
                printf("denoise run (in [%+.1f,%+.1f]): %s\n", lo, lo + (trial == 0 ? 1.0f : 2.0f), ok ? "OK" : "FAIL");
                if (ok && !outputs.empty()) {
                    stats_ch(reinterpret_cast<float*>(outputs[0].data), W, H, "out");
                }
                AiInference::get().gpu_free(d_b);
                AiInference::get().gpu_free(d_n);
                for (auto& o : outputs) AiInference::get().gpu_free(o);
            }
        }
    }
    return 0;
}
