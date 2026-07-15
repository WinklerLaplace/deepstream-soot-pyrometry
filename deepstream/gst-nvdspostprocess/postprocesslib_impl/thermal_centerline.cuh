#pragma once
#include <cuda_runtime.h>

// =================================================================
// All the geometric/dynamic content of the centerline plot. The
// static background (fixed axes, T-axis ticks/labels, "z [mm]"/"T[K]"
// labels) is built once in build_centerline_static_background().
// =================================================================

// Builds the static background (via OpenCV) at the given final
// resolution and uploads it to d_bg (device memory, pre-allocated as
// cl_w*cl_h uchar4). If CL_OUT_W/CL_OUT_H change, this regenerates
// itself automatically -- no hardcoded coordinates live outside this
// function.
void build_centerline_static_background(
    uchar4* d_bg, int cl_w, int cl_h, int margin,
    float t_min, float t_max, int n_ticks_r);

// Builds the "No data" sprite shown when the sampled column has no
// masked pixels this frame.
void build_no_data_sprite(uchar4* d_sprite, int cl_w, int cl_h);

// Finds the first/last masked row in column `col`.
// z_span[0] == -1 if there is no data this frame.
__global__ void column_span_kernel(
    const unsigned char* __restrict__ mask,
    int H, int W, int col,
    int* __restrict__ z_span);

// Draws the z-axis tick marks (geometry only) and, for every labelled
// tick (every label_stride-th), records its center position and
// physical value into the output buffers so format_z_labels_kernel /
// draw_glyphs_kernel (thermal_text) can render the actual digits.
__global__ void draw_z_ticks_kernel(
    uchar4* __restrict__ canvas,
    int canvas_w, int canvas_h, int margin,
    const int* __restrict__ z_span,
    int n_ticks, int label_stride,
    float scale_z,
    int* __restrict__ label_px_center,   // out: n_ticks/label_stride+1 centers
    int* __restrict__ label_py_baseline, // out: same count
    float* __restrict__ label_values,    // out: physical value per label
    int* __restrict__ n_labels_out);

// Draws the T(z) curve for column `col` via a DDA line draw (handles
// arbitrary slope between consecutive points).
__global__ void draw_curve_kernel(
    uchar4* __restrict__ canvas,
    const float* __restrict__ ts_raw,
    const unsigned char* __restrict__ mask,
    float ts_mean, float ts_std,
    float t_min, float t_max,
    int H, int W, int col,
    int canvas_w, int canvas_h, int margin,
    const int* __restrict__ z_span);

// Conditionally blits the "No data" sprite over the whole canvas,
// used when z_span[0] == -1 (no masked pixels this frame).
__global__ void blit_no_data_kernel(
    uchar4* __restrict__ canvas,
    const uchar4* __restrict__ sprite,
    int canvas_w, int canvas_h,
    const int* __restrict__ z_span);
    