#include "thermal_kernels.cuh"
#include "thermal_colormaps.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <cstdio>

// ─────────────────────────────────────────────────────────────
// Otsu thresholding
// ─────────────────────────────────────────────────────────────

// Each thread processes one pixel. Uses shared memory to reduce
// global atomic contention: accumulates into a per-block histogram,
// then merges into the global one with a single atomic per bin.
__global__ void otsu_histogram_kernel(
    const float*  __restrict__ g_channel,
    unsigned int* __restrict__ histogram,
    int N)
{
    __shared__ unsigned int local_hist[256];

    int tid = threadIdx.x;
    for (int i = tid; i < 256; i += blockDim.x)
        local_hist[i] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float v = fmaxf(0.f, fminf(1.f, g_channel[idx]));
        unsigned int bin = (unsigned int)(v * 255.f + 0.5f);
        atomicAdd(&local_hist[bin], 1u);
    }
    __syncthreads();

    for (int i = tid; i < 256; i += blockDim.x)
        atomicAdd(&histogram[i], local_hist[i]);
}

// Maximises inter-class variance over all 256 possible thresholds.
// Runs on the host; the histogram is only 256 ints (1 KB transfer).
int otsu_threshold_from_histogram(const unsigned int* hist, int total)
{
    double sum_all = 0.0;
    for (int i = 0; i < 256; i++) sum_all += i * hist[i];

    double sum_bg = 0.0;
    int    w_bg   = 0;
    double max_var = 0.0;
    int    thresh  = 0;

    for (int t = 0; t < 256; t++) {
        w_bg += hist[t];
        if (w_bg == 0) continue;

        int w_fg = total - w_bg;
        if (w_fg == 0) break;

        sum_bg += t * hist[t];
        double mean_bg = sum_bg / w_bg;
        double mean_fg = (sum_all - sum_bg) / w_fg;
        double diff    = mean_bg - mean_fg;
        double var_b   = (double)w_bg * w_fg * diff * diff;

        if (var_b > max_var) {
            max_var = var_b;
            thresh  = t;
        }
    }
    return thresh;
}

// Pixels above threshold_norm are foreground (255); rest are background (0).
__global__ void apply_threshold_kernel(
    const float*   __restrict__ g_norm,
    unsigned char* __restrict__ mask,
    float threshold_norm,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    mask[idx] = (g_norm[idx] > threshold_norm) ? 255 : 0;
}

// ─────────────────────────────────────────────────────────────
// Normalisation
// ─────────────────────────────────────────────────────────────

// Single-block reduction; sufficient for N=4096 (128×32).
__global__ void minmax_reduce_kernel(
    const float* __restrict__ data,
    float* __restrict__ out_min,
    float* __restrict__ out_max,
    int N)
{
    __shared__ float s_min[256];
    __shared__ float s_max[256];

    int tid = threadIdx.x;
    float lmin = FLT_MAX, lmax = -FLT_MAX;

    for (int i = tid; i < N; i += blockDim.x) {
        float v = data[i];
        lmin = fminf(lmin, v);
        lmax = fmaxf(lmax, v);
    }
    s_min[tid] = lmin;
    s_max[tid] = lmax;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        *out_min = s_min[0];
        *out_max = s_max[0];
    }
}

// Channel 1 (G) is at offset N in the CHW tensor.
__global__ void normalise_g_channel_kernel(
    const float* __restrict__ chw,
    float*       __restrict__ g_norm,
    float g_min, float g_max,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float range = g_max - g_min;
    float v = (range > 0.f) ? (chw[N + idx] - g_min) / range : 0.f;
    g_norm[idx] = fmaxf(0.f, fminf(1.f, v));
}

// channel_offset = channel_index * N; caller computes this.
__global__ void normalise_channel_kernel(
    const float* __restrict__ chw,
    float*       __restrict__ ch_norm,
    float ch_min, float ch_max,
    int channel_offset,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float range = ch_max - ch_min;
    float v = (range > 0.f) ? (chw[channel_offset + idx] - ch_min) / range : 0.f;
    ch_norm[idx] = fmaxf(0.f, fminf(1.f, v));
}

// Reverses model z-score normalisation, then remaps Kelvin → [0,1]
// over the physical range [t_min, t_max].
__global__ void normalise_ts_kernel(
    const float* __restrict__ ts_raw,
    float*       __restrict__ ts_norm,
    float ts_mean, float ts_std,
    float t_min,   float t_max,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float kelvin = ts_raw[idx] * ts_std + ts_mean;
    float v = (kelvin - t_min) / (t_max - t_min);
    ts_norm[idx] = fmaxf(0.f, fminf(1.f, v));
}

// ─────────────────────────────────────────────────────────────
// Colormaps
// ─────────────────────────────────────────────────────────────

// INFERNO colormap. Background pixels (mask == 0) are written white.
__global__ void colormap_ts_kernel(
    const float*         __restrict__ ts_norm,
    const unsigned char* __restrict__ mask,
    uchar4*              __restrict__ out,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    if (mask[idx]) {
        unsigned int bin = min((unsigned int)(ts_norm[idx] * 255.f + 0.5f), 255u);
        out[idx] = make_uchar4(kInferno[bin][0], kInferno[bin][1], kInferno[bin][2], 255);
    } else {
        out[idx] = make_uchar4(255, 255, 255, 255);
    }
}

// VIRIDIS colormap. Background pixels (mask == 0) are written white.
__global__ void colormap_channel_kernel(
    const float*         __restrict__ channel_norm,
    const unsigned char* __restrict__ mask,
    uchar4*              __restrict__ out,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    if (mask[idx]) {
        unsigned int bin = min((unsigned int)(channel_norm[idx] * 255.f + 0.5f), 255u);
        out[idx] = make_uchar4(kViridis[bin][0], kViridis[bin][1], kViridis[bin][2], 255);
    } else {
        out[idx] = make_uchar4(255, 255, 255, 255);
    }
}

// ─────────────────────────────────────────────────────────────
// Compositing
// ─────────────────────────────────────────────────────────────

__global__ void composite_panel_kernel(
    const uchar4* __restrict__ panel_data,
    uchar4*       __restrict__ canvas,
    int H, int panel_W,
    int canvas_W,
    int x_offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= panel_W || y >= H) return;

    canvas[y * canvas_W + (x_offset + x)] = panel_data[y * panel_W + x];
}

// Converts BGR → BGRA inline while copying into the canvas.
__global__ void paste_centerline_kernel(
    const unsigned char* __restrict__ cl_bgr,
    uchar4*              __restrict__ canvas,
    int H_cl, int W_cl,
    int canvas_W,
    int x_offset, int y_offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W_cl || y >= H_cl) return;

    int src = (y * W_cl + x) * 3;
    canvas[(y + y_offset) * canvas_W + (x_offset + x)] =
        make_uchar4(cl_bgr[src], cl_bgr[src + 1], cl_bgr[src + 2], 255);
}

// ─────────────────────────────────────────────────────────────
// Format conversion
// ─────────────────────────────────────────────────────────────

// BT.601 limited range. Each UV sample is the average of the four
// BGRA pixels in its 2×2 chroma block. W and H must be even
__global__ void bgra_to_nv12_kernel(
    const uchar4*  __restrict__ bgra,
    unsigned char* __restrict__ y_plane,
    unsigned char* __restrict__ uv_plane,
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    uchar4 p = bgra[y * W + x];
    float B = p.x, G = p.y, R = p.z;

    float Y = 0.257f * R + 0.504f * G + 0.098f * B + 16.f;
    y_plane[y * W + x] = (unsigned char)fminf(235.f, fmaxf(16.f, Y));

    if ((x % 2 == 0) && (y % 2 == 0)) {
        // Average the four pixels of the 2×2 chroma block
        // Guards against out-of-bounds when W or H is odd (enforced even by canvas constants)
        uchar4 p00 = bgra[ y      * W +  x     ];
        uchar4 p10 = bgra[ y      * W + (x + 1)];
        uchar4 p01 = bgra[(y + 1) * W +  x     ];
        uchar4 p11 = bgra[(y + 1) * W + (x + 1)];

        float R = (p00.z + p10.z + p01.z + p11.z) * 0.25f;
        float G = (p00.y + p10.y + p01.y + p11.y) * 0.25f;
        float B = (p00.x + p10.x + p01.x + p11.x) * 0.25f;

        float Cb = -0.148f * R - 0.291f * G + 0.439f * B + 128.f;
        float Cr =  0.439f * R  - 0.368f * G - 0.071f * B + 128.f;
        int uv_idx = (y / 2) * W + x;
        uv_plane[uv_idx]     = (unsigned char)fminf(240.f, fmaxf(16.f, Cb));
        uv_plane[uv_idx + 1] = (unsigned char)fminf(240.f, fmaxf(16.f, Cr));
    }
}
