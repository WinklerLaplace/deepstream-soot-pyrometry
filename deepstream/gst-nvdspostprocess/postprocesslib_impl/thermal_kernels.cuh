#pragma once
#include <cuda_runtime.h>

// =================================================================
// Otsu thresholding
// =================================================================
// Three-kernel pipeline, fully GPU-resident: the optimal threshold is
// computed on-device and consumed by pointer, with no host roundtrip
// between histogram construction and mask application.

// Builds a 256-bin histogram from a normalised [0,1] float array.
__global__ void otsu_histogram_kernel(
    const float*  __restrict__ g_channel,
    unsigned int* __restrict__ histogram,
    int N);

// Computes the optimal Otsu threshold entirely on GPU (single block,
// 256 threads): parallel prefix-sum scan + inter-class-variance argmax
// reduction. Writes the resulting bin index [0,255] to out_threshold_bin
// (device memory) -- the threshold never travels to the host.
__global__ void otsu_threshold_kernel(
    const unsigned int* __restrict__ hist,
    int total,
    int* __restrict__ out_threshold_bin);

// Writes a binary mask: 255 where g_norm[i] > threshold, 0 otherwise.
// threshold_bin is read from device memory (no host sync required).
__global__ void apply_threshold_kernel(
    const float*   __restrict__ g_norm,
    unsigned char* __restrict__ mask,
    const int*     __restrict__ threshold_bin,
    int N);

// =================================================================
// Normalisation
// =================================================================

// Single-block min/max reduction over N floats.
// Writes one float each to out_min and out_max.
__global__ void minmax_reduce_kernel(
    const float* __restrict__ data,
    float* __restrict__ out_min,
    float* __restrict__ out_max,
    int N);

// Extracts the G channel (index 1) from a CHW tensor and normalises
// it to [0,1] using the provided min/max (read by pointer, device memory).
__global__ void normalise_g_channel_kernel(
    const float* __restrict__ chw,
    float*       __restrict__ g_norm,
    const float* __restrict__ minmax,   // minmax[0]=min, minmax[1]=max
    int N);

// Extracts an arbitrary channel from a CHW tensor at `channel_offset`
// and normalises it to [0,1] using the provided min/max.
__global__ void normalise_channel_kernel(
    const float* __restrict__ chw,
    float*       __restrict__ ch_norm,
    const float* __restrict__ minmax,   // minmax[0]=min, minmax[1]=max
    int channel_offset,
    int N);

// Normalises raw temperature values: reverses the model's z-score
// normalisation (ts_mean, ts_std), then clamps and rescales to [0,1]
// over the physical range [t_min, t_max].
__global__ void normalise_ts_kernel(
    const float* __restrict__ ts_raw,
    float*       __restrict__ ts_norm,
    float ts_mean, float ts_std,
    float t_min,   float t_max,
    int N);

// =================================================================
// Colormaps — produce BGRA panels from normalised data
// =================================================================

// Maps normalised temperature to the INFERNO false-colour BGRA image.
// Pixels where mask[i]==0 are rendered white (background).
__global__ void colormap_ts_kernel(
    const float*         __restrict__ ts_norm,
    const unsigned char* __restrict__ mask,
    uchar4*              __restrict__ out,
    int N);

// Maps a normalised channel value to the VIRIDIS false-colour BGRA image.
// Pixels where mask[i]==0 are rendered white (background).
__global__ void colormap_channel_kernel(
    const float*         __restrict__ channel_norm,
    const unsigned char* __restrict__ mask,
    uchar4*              __restrict__ out,
    int N);

// =================================================================
// Compositing
// =================================================================

// Copies an H × panel_W BGRA panel into a larger BGRA canvas at x_offset.
// Used both for the small colormap sub-panels and for compositing
// full-resolution panels (colormap panel + centerline) side by side.
__global__ void composite_panel_kernel(
    const uchar4* __restrict__ panel_data,
    uchar4*       __restrict__ canvas,
    int H, int panel_W,
    int canvas_W,
    int x_offset);

// =================================================================
// Scaling
// =================================================================

// Bilinear BGRA scaling. Used ONLY for the Ts/B/G/R colormap panels
// (real data sampled from a small model-output tensor); the centerline
// is never scaled through this kernel -- it is built directly at final
// resolution to avoid any resampling of vector content (axes, text, curve).
__global__ void scale_bilinear_kernel(
    const uchar4* __restrict__ src, int src_w, int src_h,
    uchar4*       __restrict__ dst, int dst_w, int dst_h);

// =================================================================
// Format conversion
// =================================================================

// Converts a BGRA image to NV12 (Y plane + interleaved UV plane),
// BT.601 limited range. W and H must be even: each UV sample covers a
// 2x2 pixel block, computed as the average of the four corresponding
// BGRA pixels.
__global__ void bgra_to_nv12_kernel(
    const uchar4*  __restrict__ bgra,
    unsigned char* __restrict__ y_plane,
    unsigned char* __restrict__ uv_plane,
    int W, int H);
    