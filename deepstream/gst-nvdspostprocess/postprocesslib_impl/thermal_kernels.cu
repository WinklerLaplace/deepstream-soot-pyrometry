#include "thermal_kernels.cuh"
#include "thermal_colormaps.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <cstdio>

// =================================================================
// Otsu thresholding
// =================================================================

// Each thread processes one pixel. Uses shared memory to reduce global
// atomic contention: accumulates into a per-block histogram first, then
// merges into the global histogram with a single atomic add per bin.
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

// Single block, 256 threads (one thread per histogram bin). Maximises
// inter-class variance over all possible thresholds without ever leaving
// the device:
//   1. Hillis-Steele inclusive scan computes, for every candidate
//      threshold t, the background weight w_bg(t) and weighted sum
//      sum_bg(t) in log2(256)=8 steps.
//   2. Each thread then evaluates the inter-class variance for its own
//      bin as a candidate threshold.
//   3. A standard tree reduction finds the argmax across all 256 bins.
__global__ void otsu_threshold_kernel(
    const unsigned int* __restrict__ hist,
    int total,
    int* __restrict__ out_threshold_bin)
{
    __shared__ double s_hist[256];
    __shared__ double s_cum[256];      // running w_bg (background pixel count)
    __shared__ double s_cumsum[256];   // running sum_bg (sum of i * hist[i])

    int tid = threadIdx.x;
    s_hist[tid] = (double)hist[tid];
    __syncthreads();

    double w = s_hist[tid];
    double s = tid * s_hist[tid];
    s_cum[tid]    = w;
    s_cumsum[tid] = s;
    __syncthreads();

    for (int offset = 1; offset < 256; offset <<= 1) {
        double w_add = (tid >= offset) ? s_cum[tid - offset]    : 0.0;
        double s_add = (tid >= offset) ? s_cumsum[tid - offset] : 0.0;
        __syncthreads();
        s_cum[tid]    += w_add;
        s_cumsum[tid] += s_add;
        __syncthreads();
    }

    __shared__ double sum_all;
    if (tid == 255) sum_all = s_cumsum[255];
    __syncthreads();

    __shared__ double s_var[256];
    __shared__ int    s_idx[256];

    double w_bg = s_cum[tid];
    double w_fg = total - w_bg;
    double var_b = 0.0;
    if (w_bg > 0.0 && w_fg > 0.0) {
        double mean_bg = s_cumsum[tid] / w_bg;
        double mean_fg = (sum_all - s_cumsum[tid]) / w_fg;
        double diff = mean_bg - mean_fg;
        var_b = w_bg * w_fg * diff * diff;
    }
    s_var[tid] = var_b;
    s_idx[tid] = tid;
    __syncthreads();

    // Tree reduction for argmax
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st && s_var[tid + st] > s_var[tid]) {
            s_var[tid] = s_var[tid + st];
            s_idx[tid] = s_idx[tid + st];
        }
        __syncthreads();
    }

    if (tid == 0) *out_threshold_bin = s_idx[0];
}

// Pixels above threshold_norm are foreground (255); rest are background (0).
__global__ void apply_threshold_kernel(
    const float*   __restrict__ g_norm,
    unsigned char* __restrict__ mask,
    const int*     __restrict__ threshold_bin,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float threshold_norm = *threshold_bin / 255.f;
    mask[idx] = (g_norm[idx] > threshold_norm) ? 255 : 0;
}

// =================================================================
// Normalisation
// =================================================================

// Single-block reduction; sufficient for N=4096 (128x32 model output).
// Writes both min and max in one pass, overwriting *out_min/*out_max
// directly -- no memset needed before calling this kernel.
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

// Channel 1 (G) sits at offset N in the CHW tensor.
__global__ void normalise_g_channel_kernel(
    const float* __restrict__ chw,
    float*       __restrict__ g_norm,
    const float* __restrict__ minmax,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g_min = minmax[0], g_max = minmax[1];
    float range = g_max - g_min;
    float v = (range > 0.f) ? (chw[N + idx] - g_min) / range : 0.f;
    g_norm[idx] = fmaxf(0.f, fminf(1.f, v));
}

// channel_offset = channel_index * N; caller computes this.
__global__ void normalise_channel_kernel(
    const float* __restrict__ chw,
    float*       __restrict__ ch_norm,
    const float* __restrict__ minmax,
    int channel_offset,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float ch_min = minmax[0], ch_max = minmax[1];
    float range = ch_max - ch_min;
    float v = (range > 0.f) ? (chw[channel_offset + idx] - ch_min) / range : 0.f;
    ch_norm[idx] = fmaxf(0.f, fminf(1.f, v));
}

// Reverses the model's z-score normalisation, then remaps Kelvin to
// [0,1] over the physical range [t_min, t_max].
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

// =================================================================
// Colormaps
// =================================================================

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

// =================================================================
// Compositing
// =================================================================

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

// =================================================================
// Scaling
// =================================================================

// Standard bilinear resampling: for each destination pixel, maps back
// to source coordinates (half-pixel-center convention), then blends the
// 4 nearest source texels. Channels are handled generically via the
// CH() macro so it works uniformly across .x/.y/.z (B/G/R); alpha is
// always written as fully opaque (255), since none of the panels this
// kernel scales carry meaningful alpha.
__global__ void scale_bilinear_kernel(
    const uchar4* __restrict__ src, int src_w, int src_h,
    uchar4*       __restrict__ dst, int dst_w, int dst_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float sx = (dx + 0.5f) * src_w / dst_w - 0.5f;
    float sy = (dy + 0.5f) * src_h / dst_h - 0.5f;
    int x0 = max(0, min(src_w-1, (int)floorf(sx)));
    int y0 = max(0, min(src_h-1, (int)floorf(sy)));
    int x1 = min(src_w-1, x0+1), y1 = min(src_h-1, y0+1);
    float fx = sx - x0, fy = sy - y0;

    uchar4 c00 = src[y0*src_w+x0], c10 = src[y0*src_w+x1];
    uchar4 c01 = src[y1*src_w+x0], c11 = src[y1*src_w+x1];

    auto lerp = [](float a, float b, float t){ return a + (b-a)*t; };
    #define CH(c) (unsigned char)lerp(lerp(c00.c, c10.c, fx), lerp(c01.c, c11.c, fx), fy)
    dst[dy*dst_w+dx] = make_uchar4(CH(x), CH(y), CH(z), 255);
    #undef CH
}

// =================================================================
// Format conversion
// =================================================================

// BT.601 limited range. Each UV sample is the average of the four
// BGRA pixels in its 2x2 chroma block. W and H must be even.
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
        // Average the four pixels of the 2x2 chroma block.
        // Guarded against out-of-bounds by the caller enforcing even
        // W/H (canvas dimensions are compile-time constants chosen to
        // satisfy this).
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
