// Kagerou SDK — Unit tests for CUDA filters + pipeline.

#include "../src/pipeline.cu"
#include <cstdio>
#include <cmath>
#include <vector>

static int g_pass = 0, g_fail = 0;

static void test_pass(const char* name) {
    printf("  TEST: %-40s [PASS]\n", name);
    g_pass++;
}

static void test_fail(const char* name, const char* msg) {
    printf("  TEST: %-40s [FAIL] %s\n", name, msg);
    g_fail++;
}

// ---- Test: NV12 -> RGB -> NV12 roundtrip -----------------------------------
void test_nv12_rgb_roundtrip() {
    printf("\n[Color Conversion Tests]\n");
    const uint32_t W = 64, H = 64;

    std::vector<uint8_t> nv12(W * H + W * (H / 2));
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x)
            nv12[y * W + x] = (uint8_t)((x * 3 + y * 5) & 0xFF);
    for (uint32_t y = 0; y < H/2; ++y)
        for (uint32_t x = 0; x < W; x += 2) {
            nv12[W*H + y*W + x] = 128;
            nv12[W*H + y*W + x+1] = 128;
        }

    uint8_t *d_nv12, *d_rgb;
    KAGEROU_CUDA_CHECK(cudaMalloc(&d_nv12, nv12.size()));
    KAGEROU_CUDA_CHECK(cudaMalloc(&d_rgb, W * H * 3));
    KAGEROU_CUDA_CHECK(cudaMemcpy(d_nv12, nv12.data(), nv12.size(), cudaMemcpyHostToDevice));

    kagerou::filters::nv12_to_rgb(d_nv12, d_rgb, W, H);
    KAGEROU_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> rgb(W * H * 3);
    KAGEROU_CUDA_CHECK(cudaMemcpy(rgb.data(), d_rgb, rgb.size(), cudaMemcpyDeviceToHost));

    {
        bool ok = true;
        for (size_t i = 0; i < rgb.size(); ++i)
            if (rgb[i] > 255) { ok = false; break; }
        if (ok) test_pass("NV12->RGB: values in [0,255]");
        else test_fail("NV12->RGB: values in [0,255]", "out of range");
    }

    uint8_t* d_nv12_out;
    KAGEROU_CUDA_CHECK(cudaMalloc(&d_nv12_out, nv12.size()));
    kagerou::filters::rgb_to_nv12(d_rgb, d_nv12_out, W, H);
    KAGEROU_CUDA_CHECK(cudaDeviceSynchronize());

    {
        std::vector<uint8_t> nv12_out(nv12.size());
        KAGEROU_CUDA_CHECK(cudaMemcpy(nv12_out.data(), d_nv12_out, nv12_out.size(), cudaMemcpyDeviceToHost));
        double total_err = 0;
        for (uint32_t i = 0; i < W * H; ++i)
            total_err += abs((int)nv12_out[i] - (int)nv12[i]);
        double avg_err = total_err / (W * H);
        bool ok = (avg_err < 5.0);
        if (ok) test_pass("RGB->NV12: Y plane avg error < 5");
        else test_fail("RGB->NV12: Y plane avg error < 5", "avg error too high");
    }

    cudaFree(d_nv12); cudaFree(d_rgb); cudaFree(d_nv12_out);
}

// ---- Test: Resize ----------------------------------------------------------
void test_resize() {
    printf("\n[Resize Tests]\n");
    const uint32_t SW = 64, SH = 64, DW = 32, DH = 32;

    std::vector<uint8_t> src(SW * SH * 3);
    for (size_t i = 0; i < src.size(); ++i) src[i] = (uint8_t)(i & 0xFF);

    uint8_t *d_src, *d_dst;
    KAGEROU_CUDA_CHECK(cudaMalloc(&d_src, src.size()));
    KAGEROU_CUDA_CHECK(cudaMalloc(&d_dst, DW * DH * 3));
    KAGEROU_CUDA_CHECK(cudaMemcpy(d_src, src.data(), src.size(), cudaMemcpyHostToDevice));

    {
        kagerou::filters::resize_bilinear(d_src, d_dst, SW, SH, DW, DH, 3);
        KAGEROU_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint8_t> dst(DW * DH * 3);
        KAGEROU_CUDA_CHECK(cudaMemcpy(dst.data(), d_dst, dst.size(), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (size_t i = 0; i < dst.size(); ++i)
            if (dst[i] > 255) { ok = false; break; }
        if (ok) test_pass("Bilinear 64x64 -> 32x32");
        else test_fail("Bilinear 64x64 -> 32x32", "out of range");
    }

    {
        kagerou::filters::resize_bicubic(d_src, d_dst, SW, SH, DW, DH, 3);
        KAGEROU_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint8_t> dst(DW * DH * 3);
        KAGEROU_CUDA_CHECK(cudaMemcpy(dst.data(), d_dst, dst.size(), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (size_t i = 0; i < dst.size(); ++i)
            if (dst[i] > 255) { ok = false; break; }
        if (ok) test_pass("Bicubic 64x64 -> 32x32");
        else test_fail("Bicubic 64x64 -> 32x32", "out of range");
    }

    {
        uint8_t* d_up;
        KAGEROU_CUDA_CHECK(cudaMalloc(&d_up, 128 * 128 * 3));
        kagerou::filters::resize_bilinear(d_src, d_up, SW, SH, 128, 128, 3);
        KAGEROU_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint8_t> up(128 * 128 * 3);
        KAGEROU_CUDA_CHECK(cudaMemcpy(up.data(), d_up, up.size(), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (size_t i = 0; i < up.size(); ++i)
            if (up[i] > 255) { ok = false; break; }
        if (ok) test_pass("Bilinear 64x64 -> 128x128 (upscale)");
        else test_fail("Bilinear 64x64 -> 128x128 (upscale)", "out of range");
        cudaFree(d_up);
    }

    cudaFree(d_src); cudaFree(d_dst);
}

// ---- Test: Denoise ---------------------------------------------------------
void test_denoise() {
    printf("\n[Denoise Tests]\n");
    const uint32_t W = 64, H = 64;

    std::vector<uint8_t> src(W * H * 3);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            uint8_t val = (uint8_t)(128 + 50 * sin(x * 0.1));
            src[(y*W+x)*3+0] = val;
            src[(y*W+x)*3+1] = val;
            src[(y*W+x)*3+2] = val;
        }
    for (size_t i = 0; i < src.size(); i += 3) {
        int noise = (int)src[i] + ((i * 7) % 31) - 15;
        src[i] = src[i+1] = src[i+2] = (uint8_t)(noise < 0 ? 0 : (noise > 255 ? 255 : noise));
    }

    uint8_t *d_src, *d_dst;
    KAGEROU_CUDA_CHECK(cudaMalloc(&d_src, src.size()));
    KAGEROU_CUDA_CHECK(cudaMalloc(&d_dst, src.size()));
    KAGEROU_CUDA_CHECK(cudaMemcpy(d_src, src.data(), src.size(), cudaMemcpyHostToDevice));

    {
        kagerou::filters::denoise_bilateral(d_src, d_dst, W, H, 3, 15.0f, 25.0f, 5);
        KAGEROU_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint8_t> dst(src.size());
        KAGEROU_CUDA_CHECK(cudaMemcpy(dst.data(), d_dst, dst.size(), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (size_t i = 0; i < dst.size(); ++i)
            if (dst[i] > 255) { ok = false; break; }
        if (ok) test_pass("Bilateral denoise 5x5");
        else test_fail("Bilateral denoise 5x5", "out of range");
    }

    cudaFree(d_src); cudaFree(d_dst);
}

// ---- Test: Super Resolution ------------------------------------------------
void test_super_res() {
    printf("\n[Super Resolution Tests]\n");
    const uint32_t SW = 32, SH = 32;

    std::vector<uint8_t> src(SW * SH * 3);
    for (size_t i = 0; i < src.size(); ++i) src[i] = (uint8_t)((i * 3) & 0xFF);

    uint8_t *d_src, *d_dst;
    KAGEROU_CUDA_CHECK(cudaMalloc(&d_src, src.size()));
    KAGEROU_CUDA_CHECK(cudaMalloc(&d_dst, SW*2 * SH*2 * 3));
    KAGEROU_CUDA_CHECK(cudaMemcpy(d_src, src.data(), src.size(), cudaMemcpyHostToDevice));

    {
        kagerou::filters::super_res_2x(d_src, d_dst, SW, SH, 3, 0.5f);
        KAGEROU_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint8_t> dst(SW*2 * SH*2 * 3);
        KAGEROU_CUDA_CHECK(cudaMemcpy(dst.data(), d_dst, dst.size(), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (size_t i = 0; i < dst.size(); ++i)
            if (dst[i] > 255) { ok = false; break; }
        if (ok) test_pass("2x super-res 32x32 -> 64x64");
        else test_fail("2x super-res 32x32 -> 64x64", "out of range");
    }

    cudaFree(d_src); cudaFree(d_dst);
}

// ---- Test: Frame Blend -----------------------------------------------------
void test_frame_blend() {
    printf("\n[Frame Interpolation Tests]\n");
    const uint32_t W = 64, H = 64;
    size_t size = W * H * 3;

    std::vector<uint8_t> frame_a(size, 100);
    std::vector<uint8_t> frame_b(size, 200);

    uint8_t *d_a, *d_b, *d_out;
    KAGEROU_CUDA_CHECK(cudaMalloc(&d_a, size));
    KAGEROU_CUDA_CHECK(cudaMalloc(&d_b, size));
    KAGEROU_CUDA_CHECK(cudaMalloc(&d_out, size));
    KAGEROU_CUDA_CHECK(cudaMemcpy(d_a, frame_a.data(), size, cudaMemcpyHostToDevice));
    KAGEROU_CUDA_CHECK(cudaMemcpy(d_b, frame_b.data(), size, cudaMemcpyHostToDevice));

    {
        kagerou::filters::frame_blend(d_a, d_b, d_out, W, H, 3, 0.5f);
        KAGEROU_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint8_t> out(size);
        KAGEROU_CUDA_CHECK(cudaMemcpy(out.data(), d_out, size, cudaMemcpyDeviceToHost));
        bool ok = (out[0] >= 149 && out[0] <= 151);
        if (ok) test_pass("Frame blend alpha=0.5 (100,200) -> 150");
        else test_fail("Frame blend alpha=0.5 (100,200) -> 150", "expected ~150");
    }

    {
        kagerou::filters::frame_blend(d_a, d_b, d_out, W, H, 3, 0.0f);
        KAGEROU_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint8_t> out(size);
        KAGEROU_CUDA_CHECK(cudaMemcpy(out.data(), d_out, size, cudaMemcpyDeviceToHost));
        bool ok = (out[0] >= 99 && out[0] <= 101);
        if (ok) test_pass("Frame blend alpha=0.0 (100,200) -> 100");
        else test_fail("Frame blend alpha=0.0 (100,200) -> 100", "expected ~100");
    }

    {
        kagerou::filters::frame_blend(d_a, d_b, d_out, W, H, 3, 1.0f);
        KAGEROU_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint8_t> out(size);
        KAGEROU_CUDA_CHECK(cudaMemcpy(out.data(), d_out, size, cudaMemcpyDeviceToHost));
        bool ok = (out[0] >= 199 && out[0] <= 201);
        if (ok) test_pass("Frame blend alpha=1.0 (100,200) -> 200");
        else test_fail("Frame blend alpha=1.0 (100,200) -> 200", "expected ~200");
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
}

// ---- Test: Pipeline --------------------------------------------------------
void test_pipeline() {
    printf("\n[Pipeline Tests]\n");

    kagerou::PipelineConfig cfg;
    cfg.verbose = false;
    cfg.decoder.codec = kagerou::VideoCodec::kH264;
    cfg.decoder.max_width  = 640;
    cfg.decoder.max_height = 480;
    cfg.encoder.codec = kagerou::VideoCodec::kH264;
    cfg.encoder.width  = 640;
    cfg.encoder.height = 480;
    cfg.encoder.fps    = 30;
    cfg.encoder.bitrate_kbps = 2000;

    kagerou::Pipeline pipeline;
    kagerou::Error e = pipeline.init(cfg);
    if (e != kagerou::Error::kOk) {
        test_fail("Pipeline init + 10 frames", kagerou::error_string(e));
        return;
    }

    bool ok = true;
    for (int i = 0; i < 10; ++i) {
        std::vector<uint8_t> encoded;
        e = pipeline.process_frame(nullptr, 0, encoded);
        if (e != kagerou::Error::kOk) { ok = false; break; }
        if (encoded.empty()) { ok = false; break; }
    }
    pipeline.destroy();
    if (ok) test_pass("Pipeline init + 10 frames");
    else test_fail("Pipeline init + 10 frames", "pipeline error");
}

// ---- Test: GPU Compositor ---------------------------------------------------
void test_compositor() {
    printf("\n[Compositor Tests]\n");

    const uint32_t SRC_W = 64, SRC_H = 64;
    const uint32_t TILE_W = 64, TILE_H = 64;
    const int N = 4;
    uint32_t out_w = TILE_W * 2;
    uint32_t out_h = TILE_H * 2;

    // Create N source frames on GPU (each with a different fill value)
    std::vector<uint8_t*> d_sources(N);
    for (int i = 0; i < N; ++i) {
        size_t nv12_size = SRC_W * SRC_H * 3 / 2;
        std::vector<uint8_t> host(nv12_size, (uint8_t)(50 * (i + 1)));
        // Y plane
        memset(host.data(), 50 * (i + 1), SRC_W * SRC_H);
        // UV plane
        memset(host.data() + SRC_W * SRC_H, 128, SRC_W * SRC_H / 2);

        KAGEROU_CUDA_CHECK(cudaMalloc(&d_sources[i], nv12_size));
        KAGEROU_CUDA_CHECK(cudaMemcpy(d_sources[i], host.data(), nv12_size, cudaMemcpyHostToDevice));
    }

    uint8_t* d_out;
    KAGEROU_CUDA_CHECK(cudaMalloc(&d_out, out_w * out_h * 3 / 2));

    // Build tile array: 2x2 grid
    kagerou::filters::CompositeTile tiles[4];
    for (int i = 0; i < N; ++i) {
        tiles[i].d_src = d_sources[i];
        tiles[i].src_w = SRC_W;
        tiles[i].src_h = SRC_H;
        tiles[i].dst_x = (i % 2) * TILE_W;
        tiles[i].dst_y = (i / 2) * TILE_H;
        tiles[i].tile_w = TILE_W;
        tiles[i].tile_h = TILE_H;
        tiles[i].active = true;
    }

    kagerou::filters::composite_nv12(tiles, N, d_out, out_w, out_h);
    KAGEROU_CUDA_CHECK(cudaDeviceSynchronize());

    // Read back and verify
    std::vector<uint8_t> out_host(out_w * out_h * 3 / 2);
    KAGEROU_CUDA_CHECK(cudaMemcpy(out_host.data(), d_out, out_host.size(), cudaMemcpyDeviceToHost));

    {
        // Check each tile region has its expected value
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            uint32_t tx = (i % 2) * TILE_W;
            uint32_t ty = (i / 2) * TILE_H;
            uint8_t expected = 50 * (i + 1);
            // Check Y plane center pixel
            uint32_t cx = tx + TILE_W / 2;
            uint32_t cy = ty + TILE_H / 2;
            uint8_t actual = out_host[cy * out_w + cx];
            if (actual != expected) {
                char msg[128];
                snprintf(msg, sizeof(msg), "tile %d: expected %d got %d", i, expected, actual);
                test_fail("Compositor 2x2 grid", msg);
                ok = false;
                break;
            }
        }
        if (ok) test_pass("Compositor 2x2 grid");
    }

    // Test compute_composite_grid
    {
        int cols, rows;
        uint32_t tw, th;
        kagerou::filters::compute_composite_grid(4, 1920, 1080, cols, rows, tw, th);
        if (cols == 2 && rows == 2 && tw == 960 && th == 540)
            test_pass("compute_composite_grid(4)");
        else
            test_fail("compute_composite_grid(4)", "wrong layout");
    }

    for (auto p : d_sources) cudaFree(p);
    cudaFree(d_out);
}

// ---- Test: CUDA Graph -------------------------------------------------------
void test_cuda_graph() {
    printf("\n[CUDA Graph Tests]\n");

    kagerou::PipelineConfig cfg;
    cfg.verbose = false;
    cfg.decoder.codec = kagerou::VideoCodec::kH264;
    cfg.decoder.max_width  = 640;
    cfg.decoder.max_height = 480;
    cfg.encoder.codec = kagerou::VideoCodec::kH264;
    cfg.encoder.width  = 640;
    cfg.encoder.height = 480;
    cfg.encoder.fps    = 30;
    cfg.encoder.bitrate_kbps = 2000;
    cfg.denoise.enabled = true;
    cfg.scale.enabled = true;
    cfg.scale.target_width = 320;
    cfg.scale.target_height = 240;

    kagerou::Pipeline pipeline;
    kagerou::Error e = pipeline.init(cfg);
    if (e != kagerou::Error::kOk) {
        test_fail("CUDA graph capture", kagerou::error_string(e));
        return;
    }

    // Capture graph
    e = pipeline.capture_graph();
    if (e == kagerou::Error::kOk && pipeline.graph_captured)
        test_pass("CUDA graph capture");
    else
        test_fail("CUDA graph capture", kagerou::error_string(e));

    // Replay graph
    if (pipeline.graph_captured) {
        e = pipeline.replay_graph();
        if (e == kagerou::Error::kOk)
            test_pass("CUDA graph replay");
        else
            test_fail("CUDA graph replay", kagerou::error_string(e));
    }

    pipeline.destroy();
}

// ---- Test: Crop, Chroma Key, Background Blur, Temporal, HDR -----------------
void test_transforms() {
    printf("\n[Transform Filter Tests]\n");

    const uint32_t W = 128, H = 128;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Crop test
    {
        size_t sz = W * H * 3;
        uint8_t *d_in, *d_out;
        cudaMalloc(&d_in, sz); cudaMalloc(&d_out, 64 * 64 * 3);
        cudaMemset(d_in, 100, sz);
        kagerou::filters::crop_rgb(d_in, d_out, W, H, 64, 64, 32, 32, 3, stream);
        cudaStreamSynchronize(stream);
        std::vector<uint8_t> host(64 * 64 * 3);
        cudaMemcpy(host.data(), d_out, host.size(), cudaMemcpyDeviceToHost);
        bool ok = (host[0] == 100);
        if (ok) test_pass("crop_rgb");
        else test_fail("crop_rgb", "wrong pixel value");
        cudaFree(d_in); cudaFree(d_out);
    }

    // Chroma key test
    {
        size_t sz = W * H * 3;
        uint8_t *d_in, *d_out;
        cudaMalloc(&d_in, sz); cudaMalloc(&d_out, sz);
        // Fill with green (chroma key target)
        std::vector<uint8_t> host(sz);
        for (int i = 0; i < W * H; i++) { host[i*3+0] = 0; host[i*3+1] = 200; host[i*3+2] = 0; }
        cudaMemcpy(d_in, host.data(), sz, cudaMemcpyHostToDevice);
        kagerou::filters::chroma_key_rgb(d_in, d_out, W, H,
            60.0f, 160.0f, 0.3f, 0.2f, 0.3f, 0.0f, 0.0f, 0.0f, 1.0f, stream);
        cudaStreamSynchronize(stream);
        std::vector<uint8_t> out(sz);
        cudaMemcpy(out.data(), d_out, sz, cudaMemcpyDeviceToHost);
        // Green pixels should be keyed out (bg color = 0,0,0)
        bool ok = (out[0] == 0 && out[1] == 0 && out[2] == 0);
        if (ok) test_pass("chroma_key_rgb");
        else test_fail("chroma_key_rgb", "green not keyed");
        cudaFree(d_in); cudaFree(d_out);
    }

    // Background blur test
    {
        size_t sz = W * H * 3;
        uint8_t *d_in, *d_out;
        cudaMalloc(&d_in, sz); cudaMalloc(&d_out, sz);
        cudaMemset(d_in, 128, sz);
        kagerou::filters::bg_blur_rgb(d_in, d_out, W, H,
            W * 0.5f, H * 0.5f, 0.3f, 8.0f, stream);
        cudaStreamSynchronize(stream);
        std::vector<uint8_t> out(sz);
        cudaMemcpy(out.data(), d_out, sz, cudaMemcpyDeviceToHost);
        bool ok = (out[0] != 0);
        if (ok) test_pass("bg_blur_rgb");
        else test_fail("bg_blur_rgb", "output is black");
        cudaFree(d_in); cudaFree(d_out);
    }

    // HDR tone map test (Reinhard)
    {
        size_t sz = W * H * 3 * sizeof(float);
        uint8_t *d_in, *d_out;
        cudaMalloc(&d_in, sz); cudaMalloc(&d_out, W * H * 3);
        // Fill with HDR values (>1.0 in float)
        std::vector<float> hdr(W * H * 3, 2.5f);
        cudaMemcpy(d_in, hdr.data(), sz, cudaMemcpyHostToDevice);
        kagerou::filters::hdr_tone_map_rgb(d_in, d_out, W, H, 0, 1000.0f, stream);
        cudaStreamSynchronize(stream);
        std::vector<uint8_t> out(W * H * 3);
        cudaMemcpy(out.data(), d_out, out.size(), cudaMemcpyDeviceToHost);
        bool ok = (out[0] > 0 && out[0] < 255);
        if (ok) test_pass("hdr_tone_map_reinhard");
        else test_fail("hdr_tone_map_reinhard", "output out of range");
        cudaFree(d_in); cudaFree(d_out);
    }

    // Temporal denoise test
    {
        size_t sz = W * H * 3;
        uint8_t *d_prev, *d_cur, *d_out;
        cudaMalloc(&d_prev, sz); cudaMalloc(&d_cur, sz); cudaMalloc(&d_out, sz);
        cudaMemset(d_prev, 100, sz);
        cudaMemset(d_cur, 105, sz);
        kagerou::filters::temporal_denoise_rgb(d_prev, d_cur, d_out, W, H, 0.5f, stream);
        cudaStreamSynchronize(stream);
        std::vector<uint8_t> out(sz);
        cudaMemcpy(out.data(), d_out, sz, cudaMemcpyDeviceToHost);
        // Blended value should be between 100 and 105
        bool ok = (out[0] >= 100 && out[0] <= 105);
        if (ok) test_pass("temporal_denoise_rgb");
        else test_fail("temporal_denoise_rgb", "value out of expected range");
        cudaFree(d_prev); cudaFree(d_cur); cudaFree(d_out);
    }

    cudaStreamDestroy(stream);
}

int main() {
    printf("=== Kagerou SDK Unit Tests ===\n");

    test_nv12_rgb_roundtrip();
    test_resize();
    test_denoise();
    test_super_res();
    test_frame_blend();
    test_compositor();
    test_pipeline();
    test_cuda_graph();
    test_transforms();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
