#ifdef __noinline__
#undef __noinline__
#endif

#include <cuda_runtime.h>
#include <nvbufsurftransform.h>
#include <nvbufsurface.h>

#include <glib.h>
#include <gst/gst.h>

#ifndef THERMAL_GSTNVDSINFER_INCLUDED
#define THERMAL_GSTNVDSINFER_INCLUDED
#include <gstnvdsinfer.h>
#endif

#include "thermal_render.h"
#include "thermal_kernels.cuh"
#include <gstnvdsmeta.h>
#include <nvdspreprocess_meta.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cmath>

// ─────────────────────────────────────────────────────────────
// Physical / model constants
// ─────────────────────────────────────────────────────────────
static constexpr float TS_MEAN   = 1861.5075235004497f;
static constexpr float TS_STD    = 296.8565934989852f;
static constexpr float T_MIN     = 1500.0f;
static constexpr float T_MAX     = 2150.0f;
static constexpr float SCALE_Z   = 5.5f  / 128.0f;
static constexpr int   N_TICKS_Z = 12;
static constexpr int   N_TICKS_R = 3;

// ─────────────────────────────────────────────────────────────
// Output canvas dimensions
// ─────────────────────────────────────────────────────────────
// Panels:     4 × (128×32) BGRA  →  128 × 128  (Ts, B, G, R side-by-side)
// Centerline: 300 × 200 BGR, scaled to MODEL_H height  →  192 × 128
// Raw canvas: 320 × 128 BGRA
// Output:     scaled 4× → 1280 × 512 NV12  (must be even for NV12)
static constexpr int MODEL_H     = 128;
static constexpr int MODEL_W     = 32;
static constexpr int N_PANELS    = 4;
static constexpr int PANELS_W    = N_PANELS * MODEL_W;     // 128
static constexpr int CL_W_CPU    = 300;
static constexpr int CL_H_CPU    = 200;
static constexpr int CL_W_SCALED = 192;
static constexpr int CANVAS_W    = PANELS_W + CL_W_SCALED; // 320
static constexpr int CANVAS_H    = MODEL_H;                // 128
static constexpr int OUT_SCALE   = 4;
static constexpr int OUT_W       = CANVAS_W * OUT_SCALE;   // 1280
static constexpr int OUT_H       = CANVAS_H * OUT_SCALE;   // 512

// ─────────────────────────────────────────────────────────────
// CUDA error-check macro
// ─────────────────────────────────────────────────────────────
// Wraps any CUDA API call and prints file/line/message on failure
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t _e = (call);                                      \
        if (_e != cudaSuccess) {                                      \
            fprintf(stderr, "[CUDA] %s:%d  %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(_e));      \
        }                                                             \
    } while (0)

// ─────────────────────────────────────────────────────────────
// Persistent GPU buffer pool
// ─────────────────────────────────────────────────────────────
// Allocated once at init, reused every frame to avoid per-frame
// cudaMalloc overhead. All sizes are fixed by the canvas constants above
struct GpuBuffers {
    // Normalised float maps (MODEL_H × MODEL_W each)
    float*         g_norm          = nullptr;  // normalised G channel for Otsu
    float*         ts_norm         = nullptr;  // normalised temperature [0,1]
    float*         ch_norm[3]      = {};       // normalised B, G, R channels
    float*         d_minmax        = nullptr;  // 2 floats: [min, max] scratch
    float*         ts_host_staging = nullptr;  // host→GPU staging for Ts

    // Otsu segmentation
    unsigned int*  histogram       = nullptr;  // 256-bin histogram
    unsigned char* mask            = nullptr;  // binary foreground mask

    // Colormap panels (BGRA, MODEL_H × MODEL_W each)
    uchar4*        panel_ts        = nullptr;
    uchar4*        panel_ch[3]     = {};

    // Composite canvases
    uchar4*        canvas_raw      = nullptr;  // CANVAS_H × CANVAS_W BGRA
    uchar4*        canvas_out      = nullptr;  // OUT_H × OUT_W BGRA (4× scaled)

    // NV12 output planes (OUT_H × OUT_W)
    unsigned char* nv12_y          = nullptr;
    unsigned char* nv12_uv         = nullptr;

    // Centerline transfer buffers
    unsigned char* cl_pinned       = nullptr;  // pinned host: CL_H_CPU × CL_W_CPU × 3
    unsigned char* cl_device       = nullptr;  // device: CANVAS_H × CL_W_SCALED × 3

    // Intermediate NvBufSurface for NvBufSurfTransform write-back
    NvBufSurface*  cuda_surf       = nullptr;

    // Unused pinned buffers (reserved for future DMA path)
    unsigned char* pin_y           = nullptr;
    unsigned char* pin_uv          = nullptr;

    bool initialised = false;

    void init(int H, int W) {
        int N = H * W;
        CUDA_CHECK(cudaMalloc(&g_norm,          N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ts_norm,         N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ts_host_staging, N * sizeof(float)));
        for (int c = 0; c < 3; c++)
            CUDA_CHECK(cudaMalloc(&ch_norm[c],  N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_minmax,        2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&histogram,       256 * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&mask,            N * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&panel_ts,        N * sizeof(uchar4)));
        for (int c = 0; c < 3; c++)
            CUDA_CHECK(cudaMalloc(&panel_ch[c], N * sizeof(uchar4)));
        CUDA_CHECK(cudaMalloc(&canvas_raw,  CANVAS_H * CANVAS_W * sizeof(uchar4)));
        CUDA_CHECK(cudaMalloc(&canvas_out,  OUT_H    * OUT_W    * sizeof(uchar4)));
        CUDA_CHECK(cudaMalloc(&nv12_y,      OUT_H    * OUT_W    * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&nv12_uv,    (OUT_H/2) * OUT_W    * sizeof(unsigned char)));
        CUDA_CHECK(cudaMallocHost(&cl_pinned, CL_H_CPU * CL_W_CPU * 3 * sizeof(unsigned char)));
        CUDA_CHECK(cudaMallocHost(&pin_y,   OUT_H      * OUT_W   * sizeof(unsigned char)));
        CUDA_CHECK(cudaMallocHost(&pin_uv, (OUT_H / 2) * OUT_W   * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&cl_device, CANVAS_H * CL_W_SCALED * 3 * sizeof(unsigned char)));

        NvBufSurfaceCreateParams cp = {};
        cp.gpuId       = 0;
        cp.width       = OUT_W;
        cp.height      = OUT_H;
        cp.colorFormat = NVBUF_COLOR_FORMAT_NV12;
        cp.layout      = NVBUF_LAYOUT_PITCH;
        cp.memType     = NVBUF_MEM_DEFAULT;

        if (NvBufSurfaceCreate(&cuda_surf, 1, &cp) != 0) {
            fprintf(stderr, "[RENDER] NvBufSurfaceCreate failed\n");
            cuda_surf = nullptr;
        } else {
            NvBufSurfaceParams& p = cuda_surf->surfaceList[0];
            fprintf(stderr, "[RENDER] cuda_surf: memType=%d dataPtr=%p "
                            "pitch[0]=%u pitch[1]=%u offset[0]=%zu offset[1]=%zu\n",
                    (int)cuda_surf->memType, p.dataPtr,
                    p.planeParams.pitch[0], p.planeParams.pitch[1],
                    (size_t)p.planeParams.offset[0], (size_t)p.planeParams.offset[1]);
            fflush(stderr);
        }

        initialised = true;
    }

    void destroy() {
        cudaFree(g_norm); cudaFree(ts_norm); cudaFree(ts_host_staging);
        cudaFree(d_minmax); cudaFree(histogram); cudaFree(mask);
        cudaFree(panel_ts); cudaFree(canvas_raw); cudaFree(canvas_out);
        cudaFree(nv12_y); cudaFree(nv12_uv);
        for (int c = 0; c < 3; c++) {
            cudaFree(ch_norm[c]);
            cudaFree(panel_ch[c]);
        }
        if (cuda_surf) { NvBufSurfaceDestroy(cuda_surf); cuda_surf = nullptr; }
        cudaFreeHost(cl_pinned);
        cudaFreeHost(pin_y);
        cudaFreeHost(pin_uv);
        cudaFree(cl_device);
        initialised = false;
    }
};

static GpuBuffers g_bufs;

// ─────────────────────────────────────────────────────────────
// Async centerline thread
// ─────────────────────────────────────────────────────────────
// The centerline plot runs on CPU (OpenCV polyline, 300×200px) and
// is offloaded to a background thread so it never blocks the pipeline
//
// Protocol per frame:
//   1. Pipeline thread copies Ts and mask to g_cl_work and signals.
//   2. Worker renders the plot and stores the result in g_cl_result
//   3. Next frame, pipeline thread picks up g_cl_result and pastes
//      it into the GPU canvas (one frame of latency, acceptable)

struct CenterlineWork {
    std::vector<float>         ts;
    std::vector<unsigned char> mask;
    int H, W;
    bool ready = false;
};

struct CenterlineResult {
    std::vector<unsigned char> bgr;  // CANVAS_H × CL_W_SCALED × 3
    bool valid = false;
};

static std::mutex              g_cl_mutex;
static std::condition_variable g_cl_cv;
static CenterlineWork          g_cl_work;
static CenterlineResult        g_cl_result;
static std::atomic<bool>       g_cl_stop{false};
static std::thread             g_cl_thread;

static cv::Mat build_centerline_cpu(
    const float* Ts, const unsigned char* mask, int H, int W)
{
    // Sample the centre column of the flame for the T(z) plot
    int col = W / 2;
    std::vector<int>   z_vals;
    std::vector<float> T_vals;

    for (int y = 0; y < H; y++) {
        if (!mask[y * W + col]) continue;
        float Tk = Ts[y * W + col] * TS_STD + TS_MEAN;
        z_vals.push_back(y);
        T_vals.push_back(Tk);
    }

    const int Wc = CL_W_CPU, Hc = CL_H_CPU, M = 35;  // M = margin px
    cv::Mat canvas(Hc, Wc, CV_8UC3, cv::Scalar(255, 255, 255));

    if (z_vals.empty()) {
        cv::putText(canvas, "No data", {Wc/2 - 30, Hc/2},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,0,0}, 1);
        return canvas;
    }

    float z_min = (float)z_vals.front();
    float z_max = (float)z_vals.back();

    auto px_x = [&](float z) -> int {
        return M + (int)((z - z_min) / (z_max - z_min + 1e-8f) * (Wc - 2*M));
    };
    auto px_y = [&](float T) -> int {
        return Hc - M - (int)(((T - T_MIN) / (T_MAX - T_MIN + 1e-8f)) * (Hc - 2*M));
    };

    // Axes
    cv::line(canvas, {M, Hc-M}, {Wc-M, Hc-M}, {0,0,0}, 1);
    cv::line(canvas, {M, M},    {M, Hc-M},     {0,0,0}, 1);

    // Z-axis ticks (label every 3rd)
    for (int t = 0; t <= N_TICKS_Z; t++) {
        float z_t = z_min + t * (z_max - z_min) / N_TICKS_Z;
        int   px  = px_x(z_t);
        cv::line(canvas, {px, Hc-M}, {px, Hc-M+3}, {0,0,0}, 1);
        if (t % 3 == 0) {
            char buf[8];
            snprintf(buf, sizeof(buf), "%.1f", z_t * SCALE_Z);
            cv::putText(canvas, buf, {px-10, Hc-M+12},
                        cv::FONT_HERSHEY_SIMPLEX, 0.28, {0,0,0}, 1);
        }
    }

    // T-axis ticks
    for (int t = 0; t <= N_TICKS_R; t++) {
        float T_t = T_MIN + t * (T_MAX - T_MIN) / N_TICKS_R;
        int   py  = px_y(T_t);
        cv::line(canvas, {M-3, py}, {M, py}, {0,0,0}, 1);
        char buf[8];
        snprintf(buf, sizeof(buf), "%.0f", T_t);
        cv::putText(canvas, buf, {1, py+4},
                    cv::FONT_HERSHEY_SIMPLEX, 0.28, {0,0,0}, 1);
    }

    // Centerline curve
    for (int i = 0; i+1 < (int)z_vals.size(); i++)
        cv::line(canvas,
                 {px_x((float)z_vals[i]),   px_y(T_vals[i])},
                 {px_x((float)z_vals[i+1]), px_y(T_vals[i+1])},
                 {0, 0, 200}, 2);

    cv::putText(canvas, "z [mm]", {Wc/2-20, Hc-3}, cv::FONT_HERSHEY_SIMPLEX, 0.35, {0,0,0}, 1);
    cv::putText(canvas, "T[K]",   {1, M-5},         cv::FONT_HERSHEY_SIMPLEX, 0.35, {0,0,0}, 1);

    return canvas;
}

static void centerline_worker()
{
    while (true) {
        CenterlineWork work;
        {
            std::unique_lock<std::mutex> lk(g_cl_mutex);
            g_cl_cv.wait(lk, []{ return g_cl_work.ready || g_cl_stop.load(); });
            if (g_cl_stop.load()) return;
            work = std::move(g_cl_work);
            g_cl_work.ready = false;
        }

        cv::Mat img = build_centerline_cpu(
            work.ts.data(), work.mask.data(), work.H, work.W);

        // Scale to CANVAS_H, then pad/crop to exactly CL_W_SCALED wide
        cv::Mat img_scaled;
        int new_w = (int)(CL_W_CPU * ((float)CANVAS_H / CL_H_CPU));
        cv::resize(img, img_scaled, {new_w, CANVAS_H}, 0, 0, cv::INTER_LINEAR);

        cv::Mat img_final(CANVAS_H, CL_W_SCALED, CV_8UC3, cv::Scalar(255,255,255));
        int copy_w = std::min(new_w, CL_W_SCALED);
        img_scaled(cv::Rect(0, 0, copy_w, CANVAS_H))
            .copyTo(img_final(cv::Rect(0, 0, copy_w, CANVAS_H)));

        std::lock_guard<std::mutex> lk(g_cl_mutex);
        g_cl_result.bgr.assign(
            img_final.data,
            img_final.data + CANVAS_H * CL_W_SCALED * 3);
        g_cl_result.valid = true;
    }
}

// ─────────────────────────────────────────────────────────────
// Canvas upscale kernel  (nearest-neighbour, integer scale factor)
// ─────────────────────────────────────────────────────────────
// Nearest-neighbour is indistinguishable from bilinear at 4× for
// synthetic colormap content, and avoids the gather penalty.
__global__ void scale_canvas_kernel(
    const uchar4* __restrict__ src,
    uchar4*       __restrict__ dst,
    int src_W, int src_H,
    int dst_W, int dst_H,
    int scale)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_W || dy >= dst_H) return;

    dst[dy * dst_W + dx] = src[(dy / scale) * src_W + (dx / scale)];
}

// ─────────────────────────────────────────────────────────────
// Public init / shutdown
// ─────────────────────────────────────────────────────────────
extern "C" void ThermalRender_Init()
{
    g_bufs.init(MODEL_H, MODEL_W);
    g_cl_thread = std::thread(centerline_worker);
}

extern "C" void ThermalRender_Shutdown()
{
    g_cl_stop.store(true);
    g_cl_cv.notify_all();
    if (g_cl_thread.joinable()) g_cl_thread.join();
    g_bufs.destroy();
}

// ─────────────────────────────────────────────────────────────
// Main render function
// ─────────────────────────────────────────────────────────────
extern "C" void ThermalRender_RenderFrame(
    NvBufSurface*  surf,
    guint          batch_id,
    void*          ts_host_ptr,
    void*          chw_device_ptr,
    NvDsFrameMeta* frame_meta,
    cudaStream_t   stream)
{
    if (!g_bufs.initialised || !ts_host_ptr) return;

    const int H = MODEL_H, W = MODEL_W, N = H * W;
    const int BLOCK  = 256;
    const int GRID_N = (N + BLOCK - 1) / BLOCK;

    // 1. Copy Ts from host to GPU staging buffer
    CUDA_CHECK(cudaMemcpyAsync(
        g_bufs.ts_host_staging, ts_host_ptr,
        N * sizeof(float), cudaMemcpyHostToDevice, stream));
    const float* d_ts_raw = g_bufs.ts_host_staging;

    // 2. Input tensor pointer (already on GPU)
    const float* d_chw = reinterpret_cast<const float*>(chw_device_ptr);

    // 3. Normalise Ts
    // Reverses z-score normalisation then remaps to [0,1] over [T_MIN, T_MAX].
    normalise_ts_kernel<<<GRID_N, BLOCK, 0, stream>>>(
        d_ts_raw, g_bufs.ts_norm,
        TS_MEAN, TS_STD, T_MIN, T_MAX, N);

    // 4. Otsu segmentation mask 
    if (d_chw) {
        // 4a. Normalise G channel (index 1 in CHW, offset = N)
        float h_minmax[2];
        CUDA_CHECK(cudaMemsetAsync(g_bufs.d_minmax, 0, 2*sizeof(float), stream));
        minmax_reduce_kernel<<<1, BLOCK, 0, stream>>>(
            d_chw + N, g_bufs.d_minmax, g_bufs.d_minmax + 1, N);
        CUDA_CHECK(cudaMemcpyAsync(h_minmax, g_bufs.d_minmax,
                                   2*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        normalise_g_channel_kernel<<<GRID_N, BLOCK, 0, stream>>>(
            d_chw, g_bufs.g_norm, h_minmax[0], h_minmax[1], N);

        // 4b. Histogram to Otsu threshold 
        CUDA_CHECK(cudaMemsetAsync(g_bufs.histogram, 0, 256*sizeof(unsigned int), stream));
        otsu_histogram_kernel<<<GRID_N, BLOCK, 0, stream>>>(
            g_bufs.g_norm, g_bufs.histogram, N);

        unsigned int h_hist[256];
        CUDA_CHECK(cudaMemcpyAsync(h_hist, g_bufs.histogram,
                                   256*sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float thresh_norm = otsu_threshold_from_histogram(h_hist, N) / 255.f;
        apply_threshold_kernel<<<GRID_N, BLOCK, 0, stream>>>(
            g_bufs.g_norm, g_bufs.mask, thresh_norm, N);
    } else {
        CUDA_CHECK(cudaMemsetAsync(g_bufs.mask, 255, N, stream));
    }

    // 5. Ts colormap panel (INFERNO)
    colormap_ts_kernel<<<GRID_N, BLOCK, 0, stream>>>(
        g_bufs.ts_norm, g_bufs.mask, g_bufs.panel_ts, N);

    // 6. BGR channel colormap panels (VIRIDIS)
    // Each channel is min-max normalised independently before colouring.
    if (d_chw) {
        for (int c = 0; c < 3; c++) {
            float h_mm[2];
            minmax_reduce_kernel<<<1, BLOCK, 0, stream>>>(
                d_chw + c * N, g_bufs.d_minmax, g_bufs.d_minmax + 1, N);
            CUDA_CHECK(cudaMemcpyAsync(h_mm, g_bufs.d_minmax,
                                       2*sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            normalise_channel_kernel<<<GRID_N, BLOCK, 0, stream>>>(
                d_chw, g_bufs.ch_norm[c], h_mm[0], h_mm[1], c * N, N);

            colormap_channel_kernel<<<GRID_N, BLOCK, 0, stream>>>(
                g_bufs.ch_norm[c], g_bufs.mask, g_bufs.panel_ch[c], N);
        }
    } else {
        for (int c = 0; c < 3; c++)
            CUDA_CHECK(cudaMemsetAsync(g_bufs.panel_ch[c], 255, N * sizeof(uchar4), stream));
    }

    // 7. Composite four panels into raw canvas
    // Layout: [Ts | B | G | R] at x = 0, W, 2W, 3W
    CUDA_CHECK(cudaMemsetAsync(g_bufs.canvas_raw, 255,
                               CANVAS_H * CANVAS_W * sizeof(uchar4), stream));
    {
        dim3 blk2d(16, 16);
        dim3 grd2d((MODEL_W + 15) / 16, (MODEL_H + 15) / 16);

        composite_panel_kernel<<<grd2d, blk2d, 0, stream>>>(
            g_bufs.panel_ts, g_bufs.canvas_raw, MODEL_H, MODEL_W, CANVAS_W, 0);

        for (int c = 0; c < 3; c++)
            composite_panel_kernel<<<grd2d, blk2d, 0, stream>>>(
                g_bufs.panel_ch[c], g_bufs.canvas_raw,
                MODEL_H, MODEL_W, CANVAS_W, (c + 1) * MODEL_W);
    }

    // 8. Paste centerline from async CPU thread
    // g_cl_result holds the plot rendered during the previous frame
    // One frame of latency is acceptable
    {
        std::lock_guard<std::mutex> lk(g_cl_mutex);
        if (g_cl_result.valid) {
            std::memcpy(g_bufs.cl_pinned, g_cl_result.bgr.data(),
                        CANVAS_H * CL_W_SCALED * 3);

            CUDA_CHECK(cudaMemcpyAsync(g_bufs.cl_device, g_bufs.cl_pinned,
                                       CANVAS_H * CL_W_SCALED * 3,
                                       cudaMemcpyHostToDevice, stream));

            dim3 blk(16, 16);
            dim3 grd((CL_W_SCALED + 15) / 16, (CANVAS_H + 15) / 16);
            paste_centerline_kernel<<<grd, blk, 0, stream>>>(
                g_bufs.cl_device, g_bufs.canvas_raw,
                CANVAS_H, CL_W_SCALED, CANVAS_W,
                PANELS_W, 0);
        }
    }

    // 9. Submit next centerline job (non-blocking)
    {
        static std::vector<float>         h_ts(N);
        static std::vector<unsigned char> h_mask(N);

        CUDA_CHECK(cudaMemcpyAsync(h_ts.data(),   d_ts_raw,    N * sizeof(float),         cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_mask.data(), g_bufs.mask, N * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::lock_guard<std::mutex> lk(g_cl_mutex);
        g_cl_work = { h_ts, h_mask, H, W, true };
        g_cl_cv.notify_one();
    }

    // 10. Scale canvas 4× (nearest-neighbour)
    {
        dim3 blk(16, 16);
        dim3 grd((OUT_W + 15) / 16, (OUT_H + 15) / 16);
        scale_canvas_kernel<<<grd, blk, 0, stream>>>(
            g_bufs.canvas_raw, g_bufs.canvas_out,
            CANVAS_W, CANVAS_H, OUT_W, OUT_H, OUT_SCALE);
    }

    // 11. BGRA → NV12
    {
        dim3 blk(16, 16);
        dim3 grd((OUT_W + 15) / 16, (OUT_H + 15) / 16);
        bgra_to_nv12_kernel<<<grd, blk, 0, stream>>>(
            g_bufs.canvas_out, g_bufs.nv12_y, g_bufs.nv12_uv, OUT_W, OUT_H);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 12. Write-back via NvBufSurfTransform
    // 12a. Map cuda_surf and copy NV12 planes from device to mapped host memory
    if (NvBufSurfaceMap(g_bufs.cuda_surf, 0, -1, NVBUF_MAP_READ_WRITE) != 0) {
        fprintf(stderr, "[RENDER] NvBufSurfaceMap cuda_surf failed\n");
        return;
    }
    NvBufSurfaceSyncForCpu(g_bufs.cuda_surf, 0, -1);

    {
        NvBufSurfaceParams& my = g_bufs.cuda_surf->surfaceList[0];
        CUDA_CHECK(cudaMemcpy2DAsync(
            my.mappedAddr.addr[0], my.planeParams.pitch[0],
            g_bufs.nv12_y,  OUT_W, OUT_W, OUT_H,     cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpy2DAsync(
            my.mappedAddr.addr[1], my.planeParams.pitch[1],
            g_bufs.nv12_uv, OUT_W, OUT_W, OUT_H / 2, cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    NvBufSurfaceSyncForDevice(g_bufs.cuda_surf, 0, -1);
    NvBufSurfaceUnMap(g_bufs.cuda_surf, 0, -1);

    // 12b. Blit cuda_surf to surf[batch_id], centred
    {
        const uint32_t dst_w = surf->surfaceList[batch_id].width;
        const uint32_t dst_h = surf->surfaceList[batch_id].height;
        const uint32_t x_off = (dst_w > (uint32_t)OUT_W) ? (dst_w - OUT_W) / 2 : 0;
        const uint32_t y_off = (dst_h > (uint32_t)OUT_H) ? (dst_h - OUT_H) / 2 : 0;

        NvBufSurfTransformRect src_rect = { 0,     0,     (uint32_t)OUT_W, (uint32_t)OUT_H };
        NvBufSurfTransformRect dst_rect = { y_off, x_off, (uint32_t)OUT_W, (uint32_t)OUT_H };

        NvBufSurfTransformParams xform = {};
        xform.transform_flag   = NVBUFSURF_TRANSFORM_CROP_SRC | NVBUFSURF_TRANSFORM_CROP_DST;
        xform.transform_flip   = NvBufSurfTransform_None;
        xform.transform_filter = NvBufSurfTransformInter_Nearest;
        xform.src_rect         = &src_rect;
        xform.dst_rect         = &dst_rect;

        NvBufSurface src_wrap     = *g_bufs.cuda_surf;
        src_wrap.numFilled        = 1;

        NvBufSurfaceParams dst_params = surf->surfaceList[batch_id];
        NvBufSurface dst_wrap         = *surf;
        dst_wrap.surfaceList          = &dst_params;
        dst_wrap.numFilled            = 1;

        int ret = NvBufSurfTransform(&src_wrap, &dst_wrap, &xform);
        if (ret != 0)
            fprintf(stderr, "[RENDER] NvBufSurfTransform failed: %d\n", ret);
    }
}
