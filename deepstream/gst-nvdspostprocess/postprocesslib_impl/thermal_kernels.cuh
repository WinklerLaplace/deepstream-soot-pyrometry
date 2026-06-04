#pragma once
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────
// Otsu thresholding
// ─────────────────────────────────────────────────────────────

// Accumulates a 256-bin histogram from a normalised float array [0,1]
__global__ void otsu_histogram_kernel(
    const float*  __restrict__ g_channel,
    unsigned int* __restrict__ histogram,
    int N);

// Computes the optimal Otsu threshold bin [0,255] on the host
// from a 256-bin histogram with `total` samples
int otsu_threshold_from_histogram(const unsigned int* hist, int total);

// Writes a binary mask: 1 where g_norm[i] >= threshold_norm, 0 otherwise
__global__ void apply_threshold_kernel(
    const float*   __restrict__ g_norm,
    unsigned char* __restrict__ mask,
    float threshold_norm,
    int N);

// ─────────────────────────────────────────────────────────────
// Normalisation
// ─────────────────────────────────────────────────────────────

// Single-block min/max reduction over N floats
// Writes one float each to out_min and out_max
__global__ void minmax_reduce_kernel(
    const float* __restrict__ data,
    float* __restrict__ out_min,
    float* __restrict__ out_max,
    int N);

// Extracts the G channel (index 1) from a CHW tensor and normalises
// it to [0,1] using the provided min/max
__global__ void normalise_g_channel_kernel(
    const float* __restrict__ chw,
    float*       __restrict__ g_norm,
    float g_min, float g_max,
    int N);

// Extracts an arbitrary channel from a CHW tensor at `channel_offset`
// and normalises it to [0,1] using the provided min/max
__global__ void normalise_channel_kernel(
    const float* __restrict__ chw,
    float*       __restrict__ ch_norm,
    float ch_min, float ch_max,
    int channel_offset,
    int N);

// Normalises raw temperature values: z-scores with (ts_mean, ts_std),
// then clamps and rescales to [0,1] over [t_min, t_max]
__global__ void normalise_ts_kernel(
    const float* __restrict__ ts_raw,
    float*       __restrict__ ts_norm,
    float ts_mean, float ts_std,
    float t_min,   float t_max,
    int N);

// ─────────────────────────────────────────────────────────────
// Colormaps — produce BGRA panels from normalised data
// ─────────────────────────────────────────────────────────────

// Maps normalised temperature to a diverging false-colour BGRA image
// Pixels where mask[i]==0 are rendered as background
__global__ void colormap_ts_kernel(
    const float*         __restrict__ ts_norm,
    const unsigned char* __restrict__ mask,
    uchar4*              __restrict__ out,
    int N);

// Maps a normalised channel value to a sequential false-colour BGRA image
// Pixels where mask[i]==0 are rendered as background
__global__ void colormap_channel_kernel(
    const float*         __restrict__ channel_norm,
    const unsigned char* __restrict__ mask,
    uchar4*              __restrict__ out,
    int N);

// ─────────────────────────────────────────────────────────────
// Compositing
// ─────────────────────────────────────────────────────────────

// Copies a MODEL_H × panel_W BGRA panel into the canvas at x_offset
__global__ void composite_panel_kernel(
    const uchar4* __restrict__ panel_data,
    uchar4*       __restrict__ canvas,
    int H, int panel_W,
    int canvas_W,
    int x_offset);

// Pastes a BGR centerline image (H_cl × W_cl) into the BGRA canvas
// at (x_offset, y_offset), converting BGR → BGRA inline
__global__ void paste_centerline_kernel(
    const unsigned char* __restrict__ cl_bgr,
    uchar4*              __restrict__ canvas,
    int H_cl, int W_cl,
    int canvas_W,
    int x_offset, int y_offset);

// ─────────────────────────────────────────────────────────────
// Format conversion
// ─────────────────────────────────────────────────────────────

// Converts a BGRA image to NV12 (Y plane + interleaved UV plane)
// W and H must be even. Each UV sample covers a 2×2 pixel block,
// computed as the average of the four corresponding luma samples
__global__ void bgra_to_nv12_kernel(
    const uchar4*  __restrict__ bgra,
    unsigned char* __restrict__ y_plane,
    unsigned char* __restrict__ uv_plane,
    int W, int H);
