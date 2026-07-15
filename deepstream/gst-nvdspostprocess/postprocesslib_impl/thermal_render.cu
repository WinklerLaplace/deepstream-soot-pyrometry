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
#include "thermal_centerline.cuh"
#include "thermal_text.cuh"
#include "thermal_panels.cuh"
#include <gstnvdsmeta.h>
#include <nvdspreprocess_meta.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <atomic>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cmath>

// =================================================================
// Layout constants
// =================================================================
// Final output canvas is composed of two side-by-side panels:
//
//   [  panel Ts/B/G/R (colormaps + labels + colorbars)  ][  centerline plot  ]
//   <---------------- PANELS_OUT_W -------------------->  <---- CL_OUT_W --->
//   <------------------------------ OUT_W --------------------------------->
//
// The colormap panel itself is subdivided vertically into three bands:
//
//   PANEL_LABEL_H   - channel name labels ("Ts", "R", "G", "B")
//   PANEL_IMG_H     - the 4 colormap sub-panels, scaled up from MODEL_H x MODEL_W
//   PANEL_FOOTER_H  - "Mean: ... K" text + both colorbars with range labels
//
// Everything here is a compile-time constant so that (a) all GPU buffers
// can be pre-allocated once in GpuBuffers::init, and (b) resolution changes
// only require editing these values -- no kernel logic depends on any
// hardcoded size, every kernel receives dimensions as parameters.

static constexpr int MODEL_H  = 128;   // model output height (Ts, CHW tensor)
static constexpr int MODEL_W  = 32;    // model output width
static constexpr int N_PANELS = 4;     // Ts, R, G, B
static constexpr int PANELS_W = N_PANELS * MODEL_W; // 128: small square canvas before upscaling

static constexpr int OUT_W = 1280;   // final output resolution (chosen experimentally,
static constexpr int OUT_H = 512;    // see memoria: balance between visual quality and throughput)

static constexpr int PANEL_LABEL_H  = 40;
static constexpr int PANEL_FOOTER_H = 110;
static constexpr int PANEL_IMG_H    = OUT_H - PANEL_LABEL_H - PANEL_FOOTER_H; // 362

// The colormap panel's output width is set equal to PANEL_IMG_H (a square),
// because panels_raw (the source for scale_bilinear_kernel) is itself square
// (PANELS_W == MODEL_H == 128). Any non-square destination here would
// scale X/Y by different factors and visibly stretch the sub-panels --
// this was an actual bug caught during development; keep this equality.
static constexpr int PANELS_OUT_W  = PANEL_IMG_H;
static constexpr int PANEL_SUB_W   = PANELS_OUT_W / N_PANELS; // width of each of the 4 sub-panel slots
static constexpr int MEAN_MAX_CHARS = 4; // physical range 1500-2150 K -> always 4 digits

static constexpr int CL_OUT_W = OUT_W - PANELS_OUT_W; // centerline gets all remaining width
static constexpr int CL_OUT_H = OUT_H;

static constexpr int CL_MARGIN = 70; // centerline plot margin, in final-resolution pixels

// Glyph cell size for the dynamic-text atlas, derived from CL_OUT_H so that
// a future resolution change regenerates the atlas at the right scale
// automatically -- no separate constant to keep in sync by hand.
static constexpr int GLYPH_W = (int)(CL_OUT_H * 0.03f);
static constexpr int GLYPH_H = (int)(CL_OUT_H * 0.045f);

static constexpr int N_TICKS_Z    = 12;  // number of z-axis tick marks
static constexpr int N_TICKS_R    = 3;   // number of T-axis tick marks (static, drawn once at Init)
static constexpr int LABEL_STRIDE = 3;   // label every 3rd z-tick, to avoid overcrowding
static constexpr int N_Z_LABELS   = N_TICKS_Z / LABEL_STRIDE + 1; // 5

// Model / physical constants (fixed by the trained network and sensor range)
static constexpr float TS_MEAN = 1861.5075235004497f;
static constexpr float TS_STD  = 296.8565934989852f;
static constexpr float T_MIN   = 1500.0f;
static constexpr float T_MAX   = 2150.0f;
static constexpr float SCALE_Z = 5.5f / 128.0f; // row index -> physical z [mm]

// =================================================================
// CUDA error-check macro
// =================================================================
// Wraps a CUDA API call and prints file/line/message on failure. Used for
// all host-side CUDA calls (malloc, memcpy, memset, stream sync); kernel
// launches themselves are not wrapped since their errors surface async.
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t _e = (call);                                      \
        if (_e != cudaSuccess) {                                      \
            fprintf(stderr, "[CUDA] %s:%d  %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(_e));      \
        }                                                             \
    } while (0)

// =================================================================
// Persistent GPU buffer pool
// =================================================================
// Every buffer is allocated once in init() and reused every frame, avoiding
// the non-deterministic overhead of cudaMalloc in the per-frame hot path.
// Static content (backgrounds, glyph atlas, colorbars) is also generated
// once here via OpenCV and uploaded to the device, instead of being
// recomputed per frame.
struct GpuBuffers {
    // ---- Normalisation intermediates ----
    float* g_norm = nullptr;          // normalised G channel, used for Otsu
    float* ts_norm = nullptr;         // normalised Ts, used for the INFERNO panel
    float* ts_host_staging = nullptr; // device copy of the host-provided Ts tensor
    float* ch_norm[3] = {};           // normalised B/G/R channels
    float* d_minmax = nullptr;        // 8 floats = 4 independent [min,max] pairs:
                                       // pair 0 = G (for Otsu), pairs 1-3 = B,G,R

    // ---- Otsu segmentation ----
    unsigned int*  histogram = nullptr; // 256-bin histogram of normalised G
    unsigned char* mask      = nullptr; // binary foreground mask (255/0)
    int*           d_otsu_bin = nullptr; // GPU-resident threshold bin, written by
                                          // otsu_threshold_kernel and read directly
                                          // by apply_threshold_kernel (no host roundtrip)

    // ---- Colormap panels (small, MODEL_H x MODEL_W each) ----
    uchar4* panel_ts = nullptr, *panel_ch[3] = {};
    uchar4* panels_raw = nullptr;   // the 4 panels composited side by side, still small

    // ---- Colormap panel, scaled up + decorated ----
    uchar4* panel_img_sq    = nullptr; // panels_raw scaled to a PANEL_IMG_H square (no stretch)
    uchar4* panels_static_bg = nullptr; // precomputed background: labels, colorbars, "Mean:"/"K"
    uchar4* panels_out      = nullptr; // panels_static_bg + panel_img_sq + dynamic mean value

    // ---- Mean temperature (dynamic text stamped onto panels_out) ----
    float*  d_mean       = nullptr;
    int*    mean_codes   = nullptr; // glyph codes for the formatted mean value
    int*    mean_nchars  = nullptr;
    int*    mean_px      = nullptr; // fixed position, set once in init() to match the
    int*    mean_py      = nullptr; // gap left for it in build_panels_static_background

    // ---- Centerline (fully GPU-resident, no CPU worker thread) ----
    uchar4* cl_static_bg  = nullptr; // precomputed background: axes, T ticks/labels, "z [mm]"/"T[K]"
    uchar4* cl_no_data_sp = nullptr; // fallback sprite shown when the sampled column has no mask data
    uchar4* cl_canvas     = nullptr; // per-frame working buffer: static bg + dynamic curve/ticks/labels
    int*    cl_z_span     = nullptr; // [z_min_row, z_max_row] of the masked sampled column this frame

    // ---- Dynamic text (z-axis labels, shared glyph atlas with the mean value) ----
    unsigned char* font_atlas = nullptr; // N_GLYPHS bitmap cells, '0'-'9' and '.'
    int*   label_codes  = nullptr; // N_Z_LABELS * max_chars glyph codes
    int*   label_nchars = nullptr;
    int*   label_px     = nullptr; // per-label horizontal center, computed by draw_z_ticks_kernel
    int*   label_py     = nullptr;
    float* label_values  = nullptr; // per-label physical z value [mm], before glyph formatting
    int*   n_labels      = nullptr; // how many z-labels are valid this frame (0 if no mask data)

    // ---- Final composite + format conversion ----
    uchar4*        canvas_out = nullptr; // OUT_W x OUT_H, panels_out + cl_canvas composited
    unsigned char* nv12_y  = nullptr;
    unsigned char* nv12_uv = nullptr;

    // ---- DeepStream write-back surface ----
    NvBufSurface* cuda_surf = nullptr; // intermediate CUDA-mappable surface used to copy the
                                       // NV12 planes into a DMA buffer DeepStream can blit from
    bool initialised = false;

    void init(int H, int W) {
        int N = H * W;

        // Normalisation intermediates
        CUDA_CHECK(cudaMalloc(&g_norm, N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ts_norm, N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ts_host_staging, N*sizeof(float)));
        for (int c=0;c<3;c++) CUDA_CHECK(cudaMalloc(&ch_norm[c], N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_minmax, 8*sizeof(float)));

        // Otsu
        CUDA_CHECK(cudaMalloc(&histogram, 256*sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_otsu_bin, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&mask, N*sizeof(unsigned char)));

        // Colormap panels (small)
        CUDA_CHECK(cudaMalloc(&panel_ts, N*sizeof(uchar4)));
        for (int c=0;c<3;c++) CUDA_CHECK(cudaMalloc(&panel_ch[c], N*sizeof(uchar4)));
        CUDA_CHECK(cudaMalloc(&panels_raw, PANELS_W*MODEL_H*sizeof(uchar4)));

        // Colormap panel, scaled + decorated
        CUDA_CHECK(cudaMalloc(&panel_img_sq, PANEL_IMG_H*PANEL_IMG_H*sizeof(uchar4)));
        CUDA_CHECK(cudaMalloc(&panels_out, PANELS_OUT_W*OUT_H*sizeof(uchar4)));
        CUDA_CHECK(cudaMalloc(&panels_static_bg, PANELS_OUT_W*OUT_H*sizeof(uchar4)));
        build_panels_static_background(panels_static_bg, PANELS_OUT_W, OUT_H,
            PANEL_LABEL_H, PANEL_IMG_H, PANEL_FOOTER_H, PANEL_SUB_W, T_MIN, T_MAX);

        // Mean temperature text
        CUDA_CHECK(cudaMalloc(&d_mean, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&mean_codes, MEAN_MAX_CHARS*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&mean_nchars, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&mean_px, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&mean_py, sizeof(int)));

        // Fixed position of the dynamic mean value -- MUST match the gap
        // left between "Mean:" and "K" in build_panels_static_background:
        // mean_label_w=130, gap_w=90 -> gap spans x=[150,240], center=195.
        // y_mean = footer_y0 + mean_h - 6, with footer_y0 = PANEL_LABEL_H + PANEL_IMG_H
        // and mean_h = 0.20 * PANEL_FOOTER_H (see build_panels_static_background).
        int h_px = 20 + 130 + 45; // bar_x0 + mean_label_w + gap_w/2
        int h_py = PANEL_LABEL_H + PANEL_IMG_H + (int)(PANEL_FOOTER_H * 0.20f) - 6;
        CUDA_CHECK(cudaMemcpy(mean_px, &h_px, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mean_py, &h_py, sizeof(int), cudaMemcpyHostToDevice));

        // Centerline
        CUDA_CHECK(cudaMalloc(&cl_static_bg,  CL_OUT_W*CL_OUT_H*sizeof(uchar4)));
        CUDA_CHECK(cudaMalloc(&cl_no_data_sp, CL_OUT_W*CL_OUT_H*sizeof(uchar4)));
        CUDA_CHECK(cudaMalloc(&cl_canvas,     CL_OUT_W*CL_OUT_H*sizeof(uchar4)));
        CUDA_CHECK(cudaMalloc(&cl_z_span, 2*sizeof(int)));
        build_centerline_static_background(cl_static_bg, CL_OUT_W, CL_OUT_H, CL_MARGIN, T_MIN, T_MAX, N_TICKS_R);
        build_no_data_sprite(cl_no_data_sp, CL_OUT_W, CL_OUT_H);

        // Dynamic text (glyph atlas + z-axis label buffers)
        CUDA_CHECK(cudaMalloc(&font_atlas, N_GLYPHS*GLYPH_W*GLYPH_H*sizeof(unsigned char)));
        build_glyph_atlas(font_atlas, GLYPH_W, GLYPH_H);
        CUDA_CHECK(cudaMalloc(&label_codes,  N_Z_LABELS*6*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&label_nchars, N_Z_LABELS*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&label_px,     N_Z_LABELS*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&label_py,     N_Z_LABELS*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&label_values, N_Z_LABELS*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&n_labels, sizeof(int)));

        // Final composite + NV12 output planes
        CUDA_CHECK(cudaMalloc(&canvas_out, OUT_W*OUT_H*sizeof(uchar4)));
        CUDA_CHECK(cudaMalloc(&nv12_y, OUT_W*OUT_H*sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&nv12_uv, (OUT_H/2)*OUT_W*sizeof(unsigned char)));

        // DeepStream write-back surface (NV12, CUDA-mappable)
        NvBufSurfaceCreateParams cp = {};
        cp.gpuId = 0; cp.width = OUT_W; cp.height = OUT_H;
        cp.colorFormat = NVBUF_COLOR_FORMAT_NV12;
        cp.layout = NVBUF_LAYOUT_PITCH;
        cp.memType = NVBUF_MEM_DEFAULT;
        if (NvBufSurfaceCreate(&cuda_surf, 1, &cp) != 0) {
            fprintf(stderr, "[RENDER] NvBufSurfaceCreate failed\n");
            cuda_surf = nullptr;
        } else {
            // NvBufSurfaceCreate does not mark the buffer as "filled" by
            // itself. Without this, Map/SyncForCpu/SyncForDevice/UnMap see
            // index 0 as out of range against numFilled==0, and print
            // "Wrong buffer index (0)" once per frame even though the
            // buffer and its memory are perfectly valid.
            cuda_surf->numFilled = 1;
        }

        initialised = true;
    }

    void destroy() {
        cudaFree(g_norm); cudaFree(ts_norm); cudaFree(ts_host_staging);
        for (int c=0;c<3;c++) { cudaFree(ch_norm[c]); cudaFree(panel_ch[c]); }
        cudaFree(d_minmax);

        cudaFree(histogram); cudaFree(d_otsu_bin); cudaFree(mask);

        cudaFree(panel_ts); cudaFree(panels_raw);

        cudaFree(panel_img_sq); cudaFree(panels_out); cudaFree(panels_static_bg);

        cudaFree(d_mean); cudaFree(mean_codes); cudaFree(mean_nchars);
        cudaFree(mean_px); cudaFree(mean_py);

        cudaFree(cl_static_bg); cudaFree(cl_no_data_sp); cudaFree(cl_canvas); cudaFree(cl_z_span);

        cudaFree(font_atlas); cudaFree(label_codes); cudaFree(label_nchars);
        cudaFree(label_px); cudaFree(label_py); cudaFree(label_values); cudaFree(n_labels);

        cudaFree(canvas_out); cudaFree(nv12_y); cudaFree(nv12_uv);

        if (cuda_surf) { NvBufSurfaceDestroy(cuda_surf); cuda_surf = nullptr; }
        initialised = false;
    }
};

static GpuBuffers g_bufs;

// =================================================================
// Public init / shutdown
// =================================================================
extern "C" void ThermalRender_Init()
{
    g_bufs.init(MODEL_H, MODEL_W);
}

extern "C" void ThermalRender_Shutdown()
{
    g_bufs.destroy();
}

// =================================================================
// Main render function
// =================================================================
// Called once per frame by the DeepStream postprocessing plugin. Every
// step below is enqueued on the same CUDA stream, so ordering between
// kernels is guaranteed by launch order without any explicit host-side
// synchronization -- the only cudaStreamSynchronize in the normal path
// is the one required before the write-back memcpy at the end.
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

    // ---- 1. Copy Ts (model output) from host to a GPU staging buffer ----
    // Ts arrives via host memory (ts_host_ptr) from NvDsInferTensorMeta;
    // everything downstream operates on the device copy (d_ts_raw).
    CUDA_CHECK(cudaMemcpyAsync(
        g_bufs.ts_host_staging, ts_host_ptr,
        N * sizeof(float), cudaMemcpyHostToDevice, stream));
    const float* d_ts_raw = g_bufs.ts_host_staging;

    // ---- 2. Input CHW tensor pointer (already resident on GPU) ----
    const float* d_chw = reinterpret_cast<const float*>(chw_device_ptr);

    // ---- 3. Normalise Ts to [0,1] over the physical range ----
    // Reverses the model's z-score normalisation (T_K = Ts*std + mean),
    // then remaps to [0,1] over [T_MIN, T_MAX] for the INFERNO colormap.
    normalise_ts_kernel<<<GRID_N, BLOCK, 0, stream>>>(
        d_ts_raw, g_bufs.ts_norm,
        TS_MEAN, TS_STD, T_MIN, T_MAX, N);

    // ---- 4. Otsu segmentation mask (from the G channel of the input tensor) ----
    if (d_chw) {
        // 4a. Launch all 4 min/max reductions independently (G for Otsu,
        // plus B/G/R for the colormap panels below). They have no data
        // dependency on each other, so they are enqueued back to back
        // without waiting -- avoids 4 separate stream syncs per frame.
        minmax_reduce_kernel<<<1, BLOCK, 0, stream>>>(
            d_chw + N, g_bufs.d_minmax + 0, g_bufs.d_minmax + 1, N);        // pair 0: G (Otsu)
        for (int c = 0; c < 3; c++)
            minmax_reduce_kernel<<<1, BLOCK, 0, stream>>>(
                d_chw + c * N, g_bufs.d_minmax + 2 + 2*c, g_bufs.d_minmax + 3 + 2*c, N); // pairs 1-3: B,G,R

        // 4b. Normalise the G channel using the min/max computed above,
        // read directly from device memory (no host roundtrip).
        normalise_g_channel_kernel<<<GRID_N, BLOCK, 0, stream>>>(
            d_chw, g_bufs.g_norm, g_bufs.d_minmax + 0, N);

        // 4c. Histogram -> GPU-resident threshold -> binary mask.
        // otsu_threshold_kernel computes the optimal bin entirely on
        // device (parallel scan + argmax reduction); the resulting
        // threshold never travels to the host.
        CUDA_CHECK(cudaMemsetAsync(g_bufs.histogram, 0, 256*sizeof(unsigned int), stream));
        otsu_histogram_kernel<<<GRID_N, BLOCK, 0, stream>>>(
            g_bufs.g_norm, g_bufs.histogram, N);
        otsu_threshold_kernel<<<1, 256, 0, stream>>>(
            g_bufs.histogram, N, g_bufs.d_otsu_bin);
        apply_threshold_kernel<<<GRID_N, BLOCK, 0, stream>>>(
            g_bufs.g_norm, g_bufs.mask, g_bufs.d_otsu_bin, N);
    } else {
        // No input tensor this frame (e.g. warm-up): treat everything as foreground.
        CUDA_CHECK(cudaMemsetAsync(g_bufs.mask, 255, N, stream));
    }

    // ---- 5. Ts colormap panel (INFERNO) ----
    colormap_ts_kernel<<<GRID_N, BLOCK, 0, stream>>>(g_bufs.ts_norm, g_bufs.mask, g_bufs.panel_ts, N);

    // ---- 6. B/G/R colormap panels (VIRIDIS) ----
    // Each channel is normalised independently using the min/max computed
    // in step 4a, then colored with the same VIRIDIS LUT.
    if (d_chw) {
        for (int c = 0; c < 3; c++) {
            normalise_channel_kernel<<<GRID_N, BLOCK, 0, stream>>>(
                d_chw, g_bufs.ch_norm[c], g_bufs.d_minmax + 2 + 2*c, c * N, N);

            colormap_channel_kernel<<<GRID_N, BLOCK, 0, stream>>>(
                g_bufs.ch_norm[c], g_bufs.mask, g_bufs.panel_ch[c], N);
        }
    } else {
        for (int c = 0; c < 3; c++)
            CUDA_CHECK(cudaMemsetAsync(g_bufs.panel_ch[c], 255, N * sizeof(uchar4), stream));
    }

    // ---- 7. Composite the 4 panels into panels_raw (small, no centerline yet) ----
    // Display order is Ts, R, G, B (left to right), even though the input
    // tensor is stored in native BGR order (offset 0=B, N=G, 2N=R, confirmed
    // empirically against the Python baseline). Only the destination offset
    // changes here -- no channel data is ever reordered.
    CUDA_CHECK(cudaMemsetAsync(g_bufs.panels_raw, 255, PANELS_W*MODEL_H*sizeof(uchar4), stream));
    {
        dim3 blk(16,16), grd((MODEL_W+15)/16, (MODEL_H+15)/16);
        composite_panel_kernel<<<grd, blk, 0, stream>>>(g_bufs.panel_ts, g_bufs.panels_raw, MODEL_H, MODEL_W, PANELS_W, 0);

        static constexpr int CH_DISPLAY_ORDER[3] = {2, 1, 0}; // panel_ch index for slots R, G, B
        for (int slot = 0; slot < 3; slot++) {
            int c = CH_DISPLAY_ORDER[slot];
            composite_panel_kernel<<<grd, blk, 0, stream>>>(
                g_bufs.panel_ch[c], g_bufs.panels_raw, MODEL_H, MODEL_W, PANELS_W, (slot+1)*MODEL_W);
        }
    }

    // ---- 8. Scale the panel square onto the static background, centered ----
    CUDA_CHECK(cudaMemcpyAsync(g_bufs.panels_out, g_bufs.panels_static_bg,
        PANELS_OUT_W * OUT_H * sizeof(uchar4), cudaMemcpyDeviceToDevice, stream));
    {
        // Scale to a PANEL_IMG_H square -- same aspect ratio as the source
        // (both are square), so no anisotropic stretching occurs.
        dim3 blk(16,16), grd((PANEL_IMG_H+15)/16, (PANEL_IMG_H+15)/16);
        scale_bilinear_kernel<<<grd, blk, 0, stream>>>(
            g_bufs.panels_raw, PANELS_W, MODEL_H,
            g_bufs.panel_img_sq, PANEL_IMG_H, PANEL_IMG_H);

        // Center the square horizontally within the PANELS_OUT_W column.
        // Assumes PANEL_IMG_H <= PANELS_OUT_W; revisit if a future
        // resolution change breaks that assumption (x_center would go negative).
        const int x_center = (PANELS_OUT_W - PANEL_IMG_H) / 2;
        dim3 grd2((PANEL_IMG_H+15)/16, (PANEL_IMG_H+15)/16);
        composite_panel_kernel<<<grd2, blk, 0, stream>>>(
            g_bufs.panel_img_sq,
            g_bufs.panels_out + PANEL_LABEL_H * PANELS_OUT_W, // shift down past the label band
            PANEL_IMG_H, PANEL_IMG_H, PANELS_OUT_W, x_center);
    }

    // ---- 8b. Mean temperature within the mask, stamped onto the panel ----
    // mean_reduce_kernel averages ts_raw (de-normalised to Kelvin) only over
    // pixels where mask != 0; format_mean_kernel turns that single float
    // into glyph codes; draw_glyphs_kernel stamps them at the fixed slot
    // (mean_px, mean_py) reserved in build_panels_static_background.
    mean_reduce_kernel<<<1, BLOCK, 0, stream>>>(
        d_ts_raw, g_bufs.mask, TS_MEAN, TS_STD, N, g_bufs.d_mean);

    format_mean_kernel<<<1, 1, 0, stream>>>(
        g_bufs.d_mean, g_bufs.mean_codes, g_bufs.mean_nchars, MEAN_MAX_CHARS);

    draw_glyphs_kernel<<<1, 64, 0, stream>>>(
        g_bufs.panels_out, PANELS_OUT_W, OUT_H,
        g_bufs.font_atlas, GLYPH_W, GLYPH_H,
        g_bufs.mean_codes, g_bufs.mean_nchars,
        g_bufs.mean_px, g_bufs.mean_py,
        1, MEAN_MAX_CHARS);

    // ---- 9. Centerline: build in its own working buffer, fully GPU-resident ----
    // Copy the precomputed static background (axes, T ticks/labels) as the
    // starting point, then draw everything that varies by frame on top.
    CUDA_CHECK(cudaMemcpyAsync(g_bufs.cl_canvas, g_bufs.cl_static_bg,
        CL_OUT_W * CL_OUT_H * sizeof(uchar4), cudaMemcpyDeviceToDevice, stream));

    // Row range of the masked sampled column (z_span[0] == -1 if no data this frame)
    column_span_kernel<<<1, 256, 0, stream>>>(g_bufs.mask, MODEL_H, MODEL_W, 0, g_bufs.cl_z_span);

    CUDA_CHECK(cudaMemsetAsync(g_bufs.n_labels, 0, sizeof(int), stream));
    draw_z_ticks_kernel<<<1, N_TICKS_Z+1, 0, stream>>>(
        g_bufs.cl_canvas, CL_OUT_W, CL_OUT_H, CL_MARGIN,
        g_bufs.cl_z_span, N_TICKS_Z, LABEL_STRIDE, SCALE_Z,
        g_bufs.label_px, g_bufs.label_py, g_bufs.label_values, g_bufs.n_labels);

    // T(z) curve via a DDA-style line draw: consecutive masked rows can be
    // more than one pixel apart horizontally once z is mapped to the plot
    // width, so a simple 1-pixel-per-row draw would leave gaps.
    draw_curve_kernel<<<(MODEL_H+63)/64, 64, 0, stream>>>(
        g_bufs.cl_canvas, d_ts_raw, g_bufs.mask, TS_MEAN, TS_STD, T_MIN, T_MAX,
        MODEL_H, MODEL_W, 0, CL_OUT_W, CL_OUT_H, CL_MARGIN, g_bufs.cl_z_span);

    // Replaces the whole plot with a static "No data" sprite if the sampled
    // column had no masked pixels this frame (z_span[0] == -1).
    blit_no_data_kernel<<<(CL_OUT_W*CL_OUT_H+255)/256, 256, 0, stream>>>(
        g_bufs.cl_canvas, g_bufs.cl_no_data_sp, CL_OUT_W, CL_OUT_H, g_bufs.cl_z_span);

    // ---- 9b. Format + stamp the dynamic z-axis tick labels ----
    format_z_labels_kernel<<<1, N_Z_LABELS, 0, stream>>>(
        g_bufs.label_values, g_bufs.n_labels, g_bufs.label_codes, g_bufs.label_nchars);

    draw_glyphs_kernel<<<N_Z_LABELS, 64, 0, stream>>>(
        g_bufs.cl_canvas, CL_OUT_W, CL_OUT_H,
        g_bufs.font_atlas, GLYPH_W, GLYPH_H,
        g_bufs.label_codes, g_bufs.label_nchars,
        g_bufs.label_px, g_bufs.label_py,
        N_Z_LABELS, /*max_chars=*/6);

    // ---- 10. Final composite: colormap panel + centerline, side by side ----
    CUDA_CHECK(cudaMemsetAsync(g_bufs.canvas_out, 255, OUT_W*OUT_H*sizeof(uchar4), stream));
    {
        dim3 blk(16,16);
        dim3 grd_p((PANELS_OUT_W+15)/16, (OUT_H+15)/16);
        composite_panel_kernel<<<grd_p, blk, 0, stream>>>(
            g_bufs.panels_out, g_bufs.canvas_out, OUT_H, PANELS_OUT_W, OUT_W, 0);

        dim3 grd_c((CL_OUT_W+15)/16, (CL_OUT_H+15)/16);
        composite_panel_kernel<<<grd_c, blk, 0, stream>>>(
            g_bufs.cl_canvas, g_bufs.canvas_out, CL_OUT_H, CL_OUT_W, OUT_W, PANELS_OUT_W);
    }

    // ---- 11. BGRA -> NV12 (BT.601 limited range) ----
    {
        dim3 blk(16,16), grd((OUT_W+15)/16, (OUT_H+15)/16);
        bgra_to_nv12_kernel<<<grd, blk, 0, stream>>>(g_bufs.canvas_out, g_bufs.nv12_y, g_bufs.nv12_uv, OUT_W, OUT_H);
    }

    // ---- 12. Write-back to the DeepStream video buffer ----
    // The decoded surface (surf) is a DMA buffer not directly writable as
    // CUDA memory, so the NV12 planes are first copied into cuda_surf's
    // mapped host memory (respecting hardware alignment/pitch), then
    // blitted into surf via NvBufSurfTransform.

    // 12a. Map cuda_surf and copy the NV12 planes from device to its mapped memory.
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
    // Only explicit sync in the normal path: must wait for both plane
    // copies to finish before handing cuda_surf off to NvBufSurfTransform.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    NvBufSurfaceSyncForDevice(g_bufs.cuda_surf, 0, -1);
    NvBufSurfaceUnMap(g_bufs.cuda_surf, 0, -1);

    // 12b. Blit cuda_surf into surf[batch_id], centered (offsets are 0 if
    // the destination surface is exactly OUT_W x OUT_H; otherwise the
    // remaining area keeps whatever content the original decoded frame had --
    // see memoria for the documented limitation and considered alternatives).
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

        NvBufSurface src_wrap = *g_bufs.cuda_surf;
        src_wrap.numFilled    = 1;

        NvBufSurfaceParams dst_params = surf->surfaceList[batch_id];
        NvBufSurface dst_wrap         = *surf;
        dst_wrap.surfaceList          = &dst_params;
        dst_wrap.numFilled            = 1;

        int ret = NvBufSurfTransform(&src_wrap, &dst_wrap, &xform);
        if (ret != 0)
            fprintf(stderr, "[RENDER] NvBufSurfTransform failed: %d\n", ret);
    }
}
